const express = require('express');
const cors = require('cors');
const neo4j = require('neo4j-driver');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Neo4j Driver
const driver = neo4j.driver(
  process.env.NEO4J_URI || 'neo4j+s://f7d62c0a.databases.neo4j.io',
  neo4j.auth.basic(
    process.env.NEO4J_USER || 'neo4j',
    process.env.NEO4J_PASSWORD || 'VBRuirRqvWKo94uoeq8vJEveOivYZFpoFsh2KNH-VaY'
  )
);

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/**
 * Parse course duration string and convert to months
 * Handles formats: "1 - 3 Months", "3 Months", "40 Hours"
 * @param {string} timeString - Duration string from course
 * @returns {number|null} Duration in months, or null if unparseable
 */
function parseCourseDuration(timeString) {
  if (!timeString || typeof timeString !== 'string') return null;
  
  const match = timeString.match(/(\d+)\s*-?\s*(\d+)?\s*(Month|Hour)/i);
  if (!match) return null;
  
  const unit = match[3].toLowerCase();
  const HOURS_PER_MONTH = 40;
  
  if (unit.startsWith('month')) {
    // Take the maximum value in range for conservative estimation
    const maxMonths = match[2] ? parseInt(match[2]) : parseInt(match[1]);
    return maxMonths;
  } else if (unit.startsWith('hour')) {
    const hours = match[2] ? parseInt(match[2]) : parseInt(match[1]);
    return Math.ceil(hours / HOURS_PER_MONTH);
  }
  
  return null;
}

/**
 * Parse and normalize skill/goal input from user
 * @param {string|array} input - Comma-separated string or array
 * @returns {array} Array of normalized lowercase strings
 */
function parseSkillsInput(input) {
  if (!input) return [];
  
  if (Array.isArray(input)) {
    return input.map(s => s.trim().toLowerCase()).filter(Boolean);
  }
  
  return input.split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
}

/**
 * Convert Neo4j integer to JavaScript number safely
 * @param {object} neo4jInt - Neo4j integer object
 * @returns {number} JavaScript number
 */
function toNumber(neo4jInt) {
  return neo4jInt ? neo4j.int(neo4jInt).toNumber() : 0;
}

// ============================================================
// COURSE ROUTES
// ============================================================

/**
 * Get all courses with advanced filtering, sorting, and pagination
 * Query params:
 * - page, limit: Pagination
 * - search: Search in title, summary, skills
 * - difficulty: Filter by difficulty level
 * - organization: Filter by organization
 * - minRating: Minimum rating (0-5)
 * - minStudents: Minimum student count
 * - duration: Filter by duration (short, medium, long)
 * - sortBy: rating, students, title, duration (default: rating)
 * - sortOrder: asc, desc (default: desc)
 */
app.get('/api/courses', async (req, res) => {
  const { 
    page = 1, 
    limit = 20, 
    difficulty, 
    organization, 
    search,
    minRating,
    minStudents,
    duration,
    sortBy = 'rating',
    sortOrder = 'desc'
  } = req.query;
  
  const skip = (parseInt(page) - 1) * parseInt(limit);
  const session = driver.session();

  try {
    let query = 'MATCH (c:Course)';
    let params = { skip: neo4j.int(skip), limit: neo4j.int(limit) };
    let whereConditions = [];

    // Base relationships
    query += `
      OPTIONAL MATCH (c)-[:HAS_DIFFICULTY]->(d:Difficulty)
      OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
      OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
    `;

    // Filter by difficulty
    if (difficulty && difficulty !== 'all') {
      whereConditions.push('d.level = $difficulty');
      params.difficulty = difficulty;
    }

    // Filter by organization
    if (organization && organization !== 'all') {
      whereConditions.push('o.name = $organization');
      params.organization = organization;
    }

    // Filter by minimum rating
    if (minRating) {
      whereConditions.push('c.rating >= $minRating');
      params.minRating = parseFloat(minRating);
    }

    // Filter by minimum students
    if (minStudents) {
      whereConditions.push('c.students >= $minStudents');
      params.minStudents = neo4j.int(parseInt(minStudents));
    }

    // Collect skills before WHERE clause
    query += `
      WITH c, d, o, collect(DISTINCT toLower(s.name)) as skills
    `;

    // Filter by duration range
    if (duration && duration !== 'all') {
      const durationFilters = {
        'short': 'c.time =~ ".*[0-9]+ Hour.*" OR c.time =~ ".*1 Month.*"',
        'medium': 'c.time =~ ".*[2-4] Month.*"',
        'long': 'c.time =~ ".*[5-9] Month.*" OR c.time =~ ".*1[0-9]+ Month.*"'
      };
      if (durationFilters[duration]) {
        whereConditions.push(`(${durationFilters[duration]})`);
      }
    }

    // Search by title, summary, or skills
    if (search) {
      whereConditions.push(`(
        toLower(c.title) CONTAINS toLower($search)
        OR toLower(c.summary) CONTAINS toLower($search)
        OR any(skill IN skills WHERE skill CONTAINS toLower($search))
      )`);
      params.search = search;
    }

    // Add WHERE clause if there are conditions
    if (whereConditions.length > 0) {
      query += ' WHERE ' + whereConditions.join(' AND ');
    }

    // Determine sort order
    const validSortFields = {
      'rating': 'c.rating',
      'students': 'c.students',
      'title': 'c.title',
      'duration': 'c.time'
    };
    
    const sortField = validSortFields[sortBy] || validSortFields.rating;
    const order = sortOrder.toLowerCase() === 'asc' ? 'ASC' : 'DESC';

    query += `
      RETURN c, d.level as difficulty, o.name as organization, skills
      ORDER BY ${sortField} ${order}
      SKIP $skip LIMIT $limit
    `;

    const result = await session.run(query, params);

    const courses = result.records.map(record => {
      const course = record.get('c').properties;
      return {
        title: course.title,
        rating: course.rating ? parseFloat(course.rating) : 0,
        reviewCount: toNumber(course.review_num),
        duration: course.time,
        students: toNumber(course.students),
        url: course.url,
        summary: course.summary,
        difficulty: record.get('difficulty'),
        organization: record.get('organization'),
        skills: record.get('skills')
      };
    });

    res.json({ 
      courses, 
      page: parseInt(page), 
      limit: parseInt(limit),
      filters: {
        difficulty,
        organization,
        minRating,
        minStudents,
        duration,
        sortBy,
        sortOrder
      }
    });
  } catch (error) {
    console.error('Courses fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch courses' });
  } finally {
    await session.close();
  }
});

// Get course details by title
app.get('/api/courses/:title', async (req, res) => {
  const { title } = req.params;
  const session = driver.session();

  try {
    const result = await session.run(
      `MATCH (c:Course {title: $title})
       OPTIONAL MATCH (c)-[:HAS_DIFFICULTY]->(d:Difficulty)
       OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
       OPTIONAL MATCH (c)-[:HAS_CERTIFICATE_TYPE]->(ct:CertificateType)
       OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
       OPTIONAL MATCH (c)-[:RELATED_TO]->(related:Course)
       OPTIONAL MATCH (prereq:Course)-[:RECOMMENDS_AFTER]->(c)
       OPTIONAL MATCH (c)-[:RECOMMENDS_AFTER]->(next:Course)
       RETURN c, d.level as difficulty, o.name as organization,
              ct.type as certificateType,
              collect(DISTINCT s.name) as skills,
              collect(DISTINCT {title: related.title, rating: related.rating}) as relatedCourses,
              collect(DISTINCT {title: prereq.title, rating: prereq.rating}) as prerequisites,
              collect(DISTINCT {title: next.title, rating: next.rating}) as nextCourses`,
      { title: decodeURIComponent(title) }
    );

    if (result.records.length === 0) {
      return res.status(404).json({ error: 'Course not found' });
    }

    const record = result.records[0];
    const course = record.get('c').properties;

    res.json({
      title: course.title,
      rating: course.rating ? parseFloat(course.rating) : 0,
      reviewCount: toNumber(course.review_num),
      duration: course.time,
      students: toNumber(course.students),
      url: course.url,
      summary: course.summary,
      description: course.description,
      difficulty: record.get('difficulty'),
      organization: record.get('organization'),
      certificateType: record.get('certificateType'),
      skills: record.get('skills'),
      relatedCourses: record.get('relatedCourses').filter(c => c.title),
      prerequisites: record.get('prerequisites').filter(c => c.title),
      nextCourses: record.get('nextCourses').filter(c => c.title)
    });
  } catch (error) {
    console.error('Course detail error:', error);
    res.status(500).json({ error: 'Failed to fetch course details' });
  } finally {
    await session.close();
  }
});

// ============================================================
// KNOWLEDGE-BASED LEARNING PATH BUILDER
// ============================================================

/**
 * Rule Engine: Applies expert rules to score and rank courses
 * Implements rule-based reasoning for intelligent recommendations
 * 
 * Scoring System:
 * - Quality (0-100): Based on rating and enrollment
 * - Time Fit (±50): Duration match with user availability
 * - Goal Alignment (0-200+): Skills matching learning goals
 * - Difficulty Progression (0-30): Proper sequence in learning path
 */
class RuleEngine {
  constructor() {
    // Quality thresholds
    this.QUALITY_TIERS = {
      EXCEPTIONAL: { rating: 4.7, students: 50000, score: 100, label: 'Exceptional' },
      GREAT: { rating: 4.5, students: 10000, score: 75, label: 'Highly rated' },
      GOOD: { rating: 4.3, students: 5000, score: 50, label: 'Well-rated' },
      ACCEPTABLE: { rating: 4.0, students: 0, score: 25, label: 'Good rating' }
    };
    
    // Time fitting scores
    this.TIME_SCORES = {
      QUICK: 40,        // <= 50% of available time
      PERFECT: 20,      // <= 100% of available time
      ACCEPTABLE: -20,  // <= 150% of available time
      TOO_LONG: -50     // > 150% of available time
    };
    
    // Skill and progression multipliers
    this.SKILL_MATCH_SCORE = 50;  // Per goal skill taught
    this.PROGRESSION_SCORES = {
      BEGINNER_START: 30,
      INTERMEDIATE_MID: 25,
      ADVANCED_END: 20
    };
  }

  /**
   * Rule 1: Quality Ranking - Prioritize high-quality courses
   * @param {object} course - Course object with rating and students
   * @returns {number} Quality score (0-100)
   */
  applyRankingRule(course) {
    const { rating = 0, students = 0 } = course;
    
    // Check tiers from highest to lowest
    for (const [tierName, tier] of Object.entries(this.QUALITY_TIERS)) {
      if (rating >= tier.rating && students >= tier.students) {
        course.qualityReason = tierName === 'EXCEPTIONAL'
          ? 'Highly rated with massive enrollment'
          : tierName === 'GREAT'
          ? 'Highly rated with strong enrollment'
          : tierName === 'GOOD'
          ? 'Well-rated with solid enrollment'
          : 'Good rating';
        return tier.score;
      }
    }
    
    return 0;
  }

  /**
   * Rule 2: Prerequisite Chaining - Check course prerequisites
   * @param {object} course - Course object
   * @param {array} allCourses - All courses in consideration
   * @returns {number} Prerequisite score
   */
  applyPrerequisiteChainRule(course, allCourses) {
    // This rule can be expanded to check course prerequisites in the learning path
    // For now, we rely on difficulty progression and skill gap analysis
    return 0;
  }

  /**
   * Rule 3: Time Fit - Score based on course duration vs available time
   * @param {object} course - Course object with duration
   * @param {number} maxDurationMonths - User's available time in months
   * @returns {number} Time fit score (-50 to 40)
   */
  applyTimeFilterRule(course, maxDurationMonths) {
    if (!maxDurationMonths) return 0;
    
    const courseDuration = parseCourseDuration(course.duration);
    if (!courseDuration) return 0;
    
    const ratio = courseDuration / maxDurationMonths;
    
    if (ratio <= 0.5) {
      course.timeReason = `Fits well within your ${maxDurationMonths} month timeframe`;
      return this.TIME_SCORES.QUICK;
    } else if (ratio <= 1.0) {
      course.timeReason = `Matches your ${maxDurationMonths} month availability`;
      return this.TIME_SCORES.PERFECT;
    } else if (ratio <= 1.5) {
      course.timeReason = `Slightly longer than your ${maxDurationMonths} month timeframe`;
      return this.TIME_SCORES.ACCEPTABLE;
    } else {
      course.timeReason = `Longer duration than ideal for your timeframe`;
      return this.TIME_SCORES.TOO_LONG;
    }
  }

  /**
   * Rule 4: Goal Alignment - Prioritize courses teaching target skills
   * @param {object} course - Course object with skills array
   * @param {array} goalSkills - User's learning goal skills
   * @returns {number} Goal alignment score (0-200+)
   */
  applySkillGapRule(course, goalSkills) {
    if (!goalSkills || goalSkills.length === 0) return 0;
    
    const courseSkills = course.skills || [];
    
    // Find skills that match user's goals
    const matchedGoalSkills = courseSkills.filter(skill =>
      goalSkills.some(goalSkill => {
        const skillLower = skill.toLowerCase();
        const goalLower = goalSkill.toLowerCase();
        return skillLower.includes(goalLower) || goalLower.includes(skillLower);
      })
    );

    if (matchedGoalSkills.length > 0) {
      const score = matchedGoalSkills.length * this.SKILL_MATCH_SCORE;
      const skillsPreview = matchedGoalSkills.slice(0, 3).join(', ');
      course.goalReason = `Teaches ${matchedGoalSkills.length} skill(s) toward your goal: ${skillsPreview}`;
      return score;
    }
    
    return 0;
  }

  /**
   * Rule 5: Difficulty Progression - Ensure proper learning sequence
   * @param {object} course - Course object with difficulty level
   * @param {number} userLevel - User's experience level (1-5)
   * @param {number} position - Position in learning path (0-based)
   * @returns {number} Progression score (0-30)
   */
  applyDifficultyProgressionRule(course, userLevel, position) {
    const difficulty = (course.difficulty || '').toLowerCase();
    
    // Early path: Prefer beginner courses for less experienced users
    if (position < 3 && difficulty.includes('beginner') && userLevel <= 2) {
      course.progressionReason = 'Perfect starting point for beginners';
      return this.PROGRESSION_SCORES.BEGINNER_START;
    }
    
    // Mid path: Intermediate courses for skill building
    if (position >= 3 && position < 7 && difficulty.includes('intermediate')) {
      course.progressionReason = 'Natural progression to intermediate level';
      return this.PROGRESSION_SCORES.INTERMEDIATE_MID;
    }
    
    // Late path: Advanced courses for mastery
    if (position >= 7 && (difficulty.includes('advanced') || difficulty.includes('others'))) {
      course.progressionReason = 'Advanced topic for skill mastery';
      return this.PROGRESSION_SCORES.ADVANCED_END;
    }
    
    return 0;
  }

  /**
   * Score and rank all courses using combined rule scores
   * @param {array} courses - Array of course objects
   * @param {array} goalSkills - User's learning goals
   * @param {number} maxDurationMonths - Available time constraint
   * @param {number} userLevel - User's experience level
   * @returns {array} Sorted array of scored courses with explanations
   */
  scoreAndRank(courses, goalSkills, maxDurationMonths, userLevel) {
    const scoredCourses = courses.map((course, index) => {
      // Apply all scoring rules
      const scores = {
        quality: this.applyRankingRule(course),
        prerequisite: this.applyPrerequisiteChainRule(course, courses),
        timeFit: this.applyTimeFilterRule(course, maxDurationMonths),
        goalAlignment: this.applySkillGapRule(course, goalSkills),
        progression: this.applyDifficultyProgressionRule(course, userLevel, index)
      };
      
      const totalScore = Object.values(scores).reduce((sum, score) => sum + score, 0);
      
      // Collect all explanation reasons
      const explanations = [
        course.goalReason,
        course.qualityReason,
        course.timeReason,
        course.progressionReason,
        course.prerequisiteReason
      ].filter(Boolean);
      
      return {
        ...course,
        score: totalScore,
        scoreBreakdown: scores,
        explanation: explanations.length > 0 
          ? explanations.join(' • ') 
          : 'Matches your learning criteria'
      };
    });
    
    // Sort by score descending
    return scoredCourses.sort((a, b) => b.score - a.score);
  }
}

// ============================================================
// LEARNING PATH BUILDER ENDPOINT
// ============================================================

/**
 * Build intelligent learning pathway using 4-stage process:
 * 1. Parse user input and learning goals
 * 2. Query Neo4j graph for relevant courses
 * 3. Apply rule engine for scoring
 * 4. Build structured learning path
 */
app.get('/api/recommendations/next/:courseTitle', async (req, res) => {
  const { courseTitle } = req.params;
  const { 
    userLevel = '3',      // Default: intermediate (1=beginner, 5=expert)
    currentSkills = '',   // Comma-separated skills user has
    learningGoals = '',   // Comma-separated skills/topics
    availableTime = ''    // Time in months
  } = req.query;
  
  const session = driver.session();

  try {
    // ============================================================
    // STAGE 1: INPUT VALIDATION & PARSING
    // ============================================================
    console.log('\n[STAGE 1: User Input Processing]');
    
    // Parse and validate inputs
    const searchTerm = decodeURIComponent(courseTitle).toLowerCase().trim();
    if (!searchTerm) {
      return res.status(400).json({ error: 'Course topic is required' });
    }
    
    const userSkills = parseSkillsInput(currentSkills);
    const goalSkills = parseSkillsInput(learningGoals);
    const maxDurationMonths = availableTime ? parseInt(availableTime) : null;
    const parsedUserLevel = Math.max(1, Math.min(5, parseInt(userLevel) || 3));
    
    console.log(`Topic: "${searchTerm}"`);
    console.log(`Current Skills: ${userSkills.length > 0 ? userSkills.join(', ') : 'None specified'}`);
    console.log(`Learning Goals: ${goalSkills.length > 0 ? goalSkills.join(', ') : 'Auto-derived from topic'}`);
    console.log(`Available Time: ${maxDurationMonths ? maxDurationMonths + ' months' : 'Flexible'}`);
    console.log(`User Level: ${parsedUserLevel} (1=beginner, 5=expert)`);

    // ============================================================
    // STAGE 2: GRAPH DATABASE QUERY
    // ============================================================
    console.log('\n[STAGE 2: Neo4j Graph Reasoning]');
    
    // Query for courses with prerequisite chain information
    const query = `
      MATCH (c:Course)-[:HAS_DIFFICULTY]->(d:Difficulty)
      OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
      OPTIONAL MATCH (prereqSkill:Skill)-[:PREREQUISITE_FOR]->(s)
      OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
      OPTIONAL MATCH (prereqCourse:Course)-[:RECOMMENDS_AFTER]->(c)
      WITH c, d, o, 
           collect(DISTINCT toLower(s.name)) as skills,
           collect(DISTINCT toLower(prereqSkill.name)) as prerequisites,
           collect(DISTINCT prereqCourse.title) as prerequisiteCourses
      WHERE toLower(c.title) CONTAINS $searchTerm
      OR toLower(c.summary) CONTAINS $searchTerm
      OR any(skill IN skills WHERE skill CONTAINS $searchTerm)
      RETURN c, d.level as difficulty, o.name as organization, 
             skills, prerequisites, prerequisiteCourses
      ORDER BY c.rating DESC, c.students DESC
      LIMIT 50
    `;

    const result = await session.run(query, {
      searchTerm: searchTerm
    });
    
    console.log(`Found ${result.records.length} potential courses from graph`);

    if (result.records.length === 0) {
      console.log('⚠️  No courses found for search term');
      return res.json({
        recommendations: [],
        metadata: {
          searchTerm,
          message: 'No courses found matching your criteria. Try a different topic or broader keywords.',
          stage: 'neo4j_query',
          totalCandidates: 0
        }
      });
    }

    // ============================================================
    // STAGE 3: RULE-BASED SCORING
    // ============================================================
    console.log('\n[STAGE 3: Rule Engine Processing]');
    
    const ruleEngine = new RuleEngine();
    
    // Transform Neo4j results into course objects
    const courseCandidates = result.records.map(record => {
      try {
        const course = record.get('c').properties;
        return {
          title: course.title || 'Untitled Course',
          rating: course.rating ? parseFloat(course.rating) : 0,
          reviewCount: toNumber(course.review_num),
          duration: course.time || 'Unknown',
          students: toNumber(course.students),
          url: course.url || '',
          summary: course.summary || '',
          difficulty: record.get('difficulty') || 'Unknown',
          organization: record.get('organization') || 'Unknown',
          skills: (record.get('skills') || []).filter(Boolean),
          prerequisites: (record.get('prerequisites') || []).filter(Boolean),
          prerequisiteCourses: (record.get('prerequisiteCourses') || []).filter(Boolean)
        };
      } catch (error) {
        console.error('Error parsing course record:', error);
        return null;
      }
    }).filter(Boolean);

    console.log(`Applying rule-based reasoning to ${courseCandidates.length} courses...`);
    
    // Use search term as fallback goal if no explicit goals provided
    const effectiveGoals = goalSkills.length > 0 ? goalSkills : [searchTerm];
    
    // Apply all expert rules and get scored courses
    const scoredCourses = ruleEngine.scoreAndRank(
      courseCandidates,
      effectiveGoals,
      maxDurationMonths,
      parsedUserLevel
    );

    console.log(`Top 5 courses after rule engine:`);
    scoredCourses.slice(0, 5).forEach((course, i) => {
      console.log(`  ${i + 1}. ${course.title} (Score: ${course.score})`);
    });

    // ============================================================
    // STAGE 4: LEARNING PATH CONSTRUCTION
    // ============================================================
    console.log('\n[STAGE 4: Building Learning Path Visualization]');
    
    const MAX_PATH_LENGTH = 10;
    const topCourses = scoredCourses.slice(0, MAX_PATH_LENGTH);
    
    // Helper function to categorize courses by difficulty
    const categorizeCoursesByDifficulty = (courses) => {
      const categories = { beginner: [], intermediate: [], advanced: [] };
      
      courses.forEach(course => {
        const diff = (course.difficulty || '').toLowerCase();
        if (diff.includes('beginner') || diff.includes('introductory')) {
          categories.beginner.push(course);
        } else if (diff.includes('intermediate') || diff.includes('mixed')) {
          categories.intermediate.push(course);
        } else if (diff.includes('advanced') || !diff.includes('beginner')) {
          categories.advanced.push(course);
        }
      });
      
      return categories;
    };
    
    const coursesByDifficulty = categorizeCoursesByDifficulty(topCourses);

    // Build learning path based on user level
    const buildLearningPath = (level, categories) => {
      const { beginner, intermediate, advanced } = categories;
      
      // Define path composition by user level
      const pathStrategies = {
        1: [4, 4, 2],  // Beginner: Focus on fundamentals
        2: [4, 4, 2],  // Still beginner-focused
        3: [2, 5, 3],  // Intermediate: Balanced approach
        4: [0, 5, 5],  // Advanced: Skip basics
        5: [0, 5, 5]   // Expert: Advanced focus
      };
      
      const strategy = pathStrategies[level] || pathStrategies[3];
      const [beginnerCount, intermediateCount, advancedCount] = strategy;
      
      return [
        ...beginner.slice(0, beginnerCount),
        ...intermediate.slice(0, intermediateCount),
        ...advanced.slice(0, advancedCount)
      ];
    };
    
    let learningPath = buildLearningPath(parsedUserLevel, coursesByDifficulty);
    
    // Fallback: Use top scored courses if difficulty filtering yields nothing
    if (learningPath.length === 0) {
      console.log('⚠️  No difficulty-filtered courses, using top scored courses');
      learningPath = topCourses;
    }

    // Helper function to build course connections in the path
    const buildCourseConnections = (course, prevCourse) => {
      if (!prevCourse) return [];
      
      // Check for skill prerequisites
      const sharedSkills = (course.prerequisites || []).filter(prereq =>
        (prevCourse.skills || []).some(skill => 
          skill.includes(prereq) || prereq.includes(skill)
        )
      );
      
      if (sharedSkills.length > 0) {
        return [{
          from: prevCourse.title,
          type: 'prerequisite',
          reason: `Requires skills from previous course: ${sharedSkills.slice(0, 2).join(', ')}`
        }];
      }
      
      return [{
        from: prevCourse.title,
        type: 'progression',
        reason: 'Next step in learning path'
      }];
    };
    
    // Build final recommendations with enriched metadata
    const recommendations = learningPath.map((course, index) => {
      const connections = buildCourseConnections(
        course, 
        index > 0 ? learningPath[index - 1] : null
      );
      
      return {
        step: index + 1,
        title: course.title,
        rating: course.rating,
        reviewCount: course.reviewCount,
        duration: course.duration,
        students: course.students,
        url: course.url,
        summary: course.summary,
        difficulty: course.difficulty,
        organization: course.organization,
        skills: course.skills.slice(0, 5), // Limit to top 5 for brevity
        score: course.score,
        explanation: course.explanation,
        connections,
        reason: index === 0 
          ? '🎯 Starting Point: ' + course.explanation
          : `📚 Step ${index + 1}: ` + course.explanation
      };
    });

    console.log(`✅ Built learning path with ${recommendations.length} courses`);
    
    // Return structured response with comprehensive metadata
    res.json({
      success: true,
      recommendations,
      metadata: {
        query: {
          searchTerm,
          currentSkills: userSkills,
          learningGoals: effectiveGoals,
          availableTime: maxDurationMonths ? `${maxDurationMonths} months` : 'Flexible',
          userLevel: parsedUserLevel
        },
        results: {
          totalCandidates: courseCandidates.length,
          pathLength: recommendations.length,
          avgScore: recommendations.length > 0
            ? Math.round(recommendations.reduce((sum, c) => sum + c.score, 0) / recommendations.length)
            : 0
        },
        pipeline: {
          stage1: 'Input parsing and validation',
          stage2: `Retrieved ${result.records.length} courses from Neo4j`,
          stage3: `Scored ${courseCandidates.length} courses using rule engine`,
          stage4: `Built ${recommendations.length}-step learning path`
        }
      }
    });
  } catch (error) {
    console.error('❌ Learning pathway error:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to build learning pathway',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  } finally {
    await session.close();
  }
});

// ============================================================
// END OF LEARNING PATH BUILDER
// ============================================================

// ============================================================
// SEARCH AND FILTER ROUTES
// ============================================================

// Get all organizations
app.get('/api/organizations', async (req, res) => {
  const session = driver.session();
  try {
    const result = await session.run(
      `MATCH (o:Organization)<-[:PROVIDED_BY]-(c:Course)
       RETURN o.name as name, count(c) as courseCount
       ORDER BY courseCount DESC`
    );

    const organizations = result.records.map(record => ({
      name: record.get('name'),
      courseCount: record.get('courseCount').toNumber()
    }));

    res.json({ organizations });
  } catch (error) {
    console.error('Organizations error:', error);
    res.status(500).json({ error: 'Failed to fetch organizations' });
  } finally {
    await session.close();
  }
});

// Get all skills
app.get('/api/skills', async (req, res) => {
  const session = driver.session();
  try {
    const result = await session.run(
      `MATCH (s:Skill)<-[:OFFERS_SKILL]-(c:Course)
       RETURN s.name as name, count(c) as courseCount
       ORDER BY courseCount DESC
       `
    );

    const skills = result.records.map(record => ({
      name: record.get('name'),
      courseCount: record.get('courseCount').toNumber()
    }));

    res.json({ skills });
  } catch (error) {
    console.error('Skills error:', error);
    res.status(500).json({ error: 'Failed to fetch skills' });
  } finally {
    await session.close();
  }
});

// ============================================================
// SERVER STARTUP
// ============================================================

app.get('/', (req, res) => {
  res.json({
    message: 'Course Recommendation API',
    version: '2.0.0',
    endpoints: {
      courses: '/api/courses',
      courseDetails: '/api/courses/:title',
      learningPath: '/api/recommendations/next/:courseTitle',
      organizations: '/api/organizations',
      skills: '/api/skills'
    }
  });
});

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('Shutting down gracefully...');
  await driver.close();
  process.exit(0);
});

app.listen(PORT, () => {
  console.log(`\n🚀 Server running on port ${PORT}`);
  console.log(`📊 Neo4j connected to: ${process.env.NEO4J_URI || 'default URI'}`);
  console.log(`\nAPI Endpoints:`);
  console.log(`  - Courses: http://localhost:${PORT}/api/courses`);
  console.log(`  - Learning Path: http://localhost:${PORT}/api/recommendations/next/:topic`);
  console.log(`  - Organizations: http://localhost:${PORT}/api/organizations`);
  console.log(`  - Skills: http://localhost:${PORT}/api/skills`);
});
