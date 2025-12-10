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
// COURSE ROUTES
// ============================================================

// Get all courses with pagination
app.get('/api/courses', async (req, res) => {
  const { page = 1, limit = 20, difficulty, organization, search } = req.query;
  const skip = (parseInt(page) - 1) * parseInt(limit);

  const session = driver.session();
  try {
    let query = 'MATCH (c:Course)';
    let params = { skip: neo4j.int(skip), limit: neo4j.int(limit) };

    // Filter by difficulty
    if (difficulty) {
      query += '-[:HAS_DIFFICULTY]->(d:Difficulty {level: $difficulty})';
      params.difficulty = difficulty;
    }

    // Filter by organization
    if (organization) {
      query += '-[:PROVIDED_BY]->(o:Organization {name: $organization})';
      params.organization = organization;
    }

    query += `
      OPTIONAL MATCH (c)-[:HAS_DIFFICULTY]->(d:Difficulty)
      OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
      OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
      WITH c, d, o, collect(DISTINCT toLower(s.name)) as skills
    `;

    // Search by title, summary, or skills
    if (search) {
      query += `
        WHERE toLower(c.title) CONTAINS toLower($search)
        OR toLower(c.summary) CONTAINS toLower($search)
        OR any(skill IN skills WHERE skill CONTAINS toLower($search))
      `;
      params.search = search;
    }

    query += `
      RETURN c, d.level as difficulty, o.name as organization, skills
      ORDER BY c.rating DESC
      SKIP $skip LIMIT $limit
    `;

    const result = await session.run(query, params);

    const courses = result.records.map(record => {
      const course = record.get('c').properties;
      return {
        title: course.title,
        rating: course.rating ? parseFloat(course.rating) : 0,
        reviewCount: course.review_num ? neo4j.int(course.review_num).toNumber() : 0,
        duration: course.time,
        students: course.students ? neo4j.int(course.students).toNumber() : 0,
        url: course.url,
        summary: course.summary,
        difficulty: record.get('difficulty'),
        organization: record.get('organization'),
        skills: record.get('skills')
      };
    });

    res.json({ courses, page: parseInt(page), limit: parseInt(limit) });
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
      reviewCount: course.review_num ? neo4j.int(course.review_num).toNumber() : 0,
      duration: course.time,
      students: course.students ? neo4j.int(course.students).toNumber() : 0,
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
// RECOMMENDATION ROUTES
// ============================================================

// Helper function to parse course duration in months
function parseCourseDuration(timeString) {
  if (!timeString) return null;
  
  // Format: "1 - 3 Months" or "3 Months" or "1 Month"
  const match = timeString.match(/(\d+)\s*-?\s*(\d+)?\s*(Month|Hour)/i);
  if (!match) return null;
  
  const unit = match[3].toLowerCase();
  if (unit.startsWith('month')) {
    // If range like "1 - 3", take the higher number
    const maxMonths = match[2] ? parseInt(match[2]) : parseInt(match[1]);
    return maxMonths;
  } else if (unit.startsWith('hour')) {
    // Convert hours to months (assume 40 hours = 1 month)
    const hours = match[2] ? parseInt(match[2]) : parseInt(match[1]);
    return Math.ceil(hours / 40);
  }
  
  return null;
}

// Get personalized recommendations based on user level and interests
app.get('/api/recommendations/personalized', async (req, res) => {
  const { skillLevel, interests } = req.query;

  if (!interests) {
    return res.status(400).json({ error: 'Interests are required' });
  }

  const session = driver.session();

  try {
    // Parse interests (comma-separated)
    const interestList = interests.split(',').map(i => i.trim().toLowerCase()).filter(i => i);

    // Define rules for each user level
    const levelRules = {
      '1': { // Very Beginner - Most strict for quality
        allowedDifficulties: ['Beginner'],
        minEnrollment: 30000,
        maxDurationMonths: 1,
        description: 'Beginner courses with proven track record'
      },
      '2': { // Beginner
        allowedDifficulties: ['Beginner'],
        minEnrollment: 10000,
        maxDurationMonths: 2,
        description: 'Beginner courses only'
      },
      '3': { // Intermediate
        allowedDifficulties: ['Intermediate', 'Others'],
        minEnrollment: 5000,
        maxDurationMonths: 4,
        description: 'Intermediate level with solid foundations'
      },
      '4': { // Upper Intermediate
        allowedDifficulties: ['Intermediate', 'Others'],
        minEnrollment: 2000,
        maxDurationMonths: 10,
        description: 'Intermediate to Advanced challenges'
      },
      '5': { // Advanced/Expert - Most relaxed for variety
        allowedDifficulties: ['Intermediate', 'Others'],
        minEnrollment: 500,
        maxDurationMonths: null,
        description: 'Advanced level - Intermediate and specialized courses'
      }
    };

    // If no skill level provided, search all levels
    if (!skillLevel || skillLevel === '') {
      const query = `
        MATCH (c:Course)
        OPTIONAL MATCH (c)-[:HAS_DIFFICULTY]->(d:Difficulty)
        OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
        OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
        WITH c, d, o, collect(DISTINCT toLower(s.name)) as skills
        WHERE any(interest IN $interests WHERE 
          any(skill IN skills WHERE skill CONTAINS interest)
          OR toLower(c.title) CONTAINS interest
          OR toLower(c.summary) CONTAINS interest
        )
        RETURN c, d.level as difficulty, o.name as organization, skills
        ORDER BY c.rating DESC, c.students DESC
        LIMIT 20
      `;

      const result = await session.run(query, { interests: interestList });

      const recommendations = result.records.map(record => {
        const course = record.get('c').properties;
        const skills = record.get('skills');
        
        const matchedSkills = interestList.filter(interest => 
          skills.some(skill => skill.includes(interest))
        );

        const matchedInTitle = interestList.filter(interest => 
          course.title.toLowerCase().includes(interest)
        );

        const allMatches = [...new Set([...matchedSkills, ...matchedInTitle])];

        return {
          title: course.title,
          rating: course.rating ? parseFloat(course.rating) : 0,
          reviewCount: course.review_num ? neo4j.int(course.review_num).toNumber() : 0,
          duration: course.time,
          students: course.students ? neo4j.int(course.students).toNumber() : 0,
          url: course.url,
          summary: course.summary,
          difficulty: record.get('difficulty') || 'Unknown',
          organization: record.get('organization'),
          skills: skills,
          reason: allMatches.length > 0 
            ? `Teaches: ${allMatches.join(', ')}`
            : `Matches your interests`
        };
      });

      return res.json({ recommendations: recommendations.slice(0, 10) });
    }

    // Get rules for the selected level
    const rules = levelRules[skillLevel];
    if (!rules) {
      return res.status(400).json({ error: 'Invalid skill level' });
    }

    // Build query based on user level rules
    const query = `
      MATCH (c:Course)-[:HAS_DIFFICULTY]->(d:Difficulty)
      WHERE d.level IN $allowedDifficulties
      AND c.students >= $minEnrollment
      OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
      OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
      WITH c, d, o, collect(DISTINCT toLower(s.name)) as skills
      WHERE any(interest IN $interests WHERE 
        any(skill IN skills WHERE skill CONTAINS interest)
        OR toLower(c.title) CONTAINS interest
        OR toLower(c.summary) CONTAINS interest
      )
      RETURN c, d.level as difficulty, o.name as organization, skills
      ORDER BY 
        CASE d.level
          WHEN $primaryDifficulty THEN 0
          ELSE 1
        END,
        c.rating DESC,
        c.students DESC
      LIMIT 20
    `;

    const result = await session.run(query, {
      allowedDifficulties: rules.allowedDifficulties,
      minEnrollment: neo4j.int(rules.minEnrollment),
      primaryDifficulty: rules.allowedDifficulties[0],
      interests: interestList
    });

    // Filter by duration if maxDurationMonths is set
    let filteredRecords = result.records;
    if (rules.maxDurationMonths !== null) {
      filteredRecords = result.records.filter(record => {
        const course = record.get('c').properties;
        const durationMonths = parseCourseDuration(course.time);
        return durationMonths === null || durationMonths <= rules.maxDurationMonths;
      });
    }

    // Check if we have at least 1 recommendation
    if (filteredRecords.length < 1) {
      // Progressive fallback: Try enrollment reduction first (60% → 40% → 20% → 10%)
      // then relax duration (150% → 200% → 300% → no limit)
      const fallbackPercentages = [0.6, 0.4, 0.2, 0.1]; // 60%, 40%, 20%, 10%
      const durationMultipliers = [1.5, 2, 3, null]; // 150%, 200%, 300%, no limit
      
      // Try all combinations until we find at least 1 result
      for (const percentage of fallbackPercentages) {
        for (const durationMultiplier of durationMultipliers) {
          const fallbackQuery = `
            MATCH (c:Course)-[:HAS_DIFFICULTY]->(d:Difficulty)
            WHERE d.level IN $allowedDifficulties
            AND c.students >= $fallbackEnrollment
            OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
            OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
            WITH c, d, o, collect(DISTINCT toLower(s.name)) as skills
            WHERE any(interest IN $interests WHERE 
              any(skill IN skills WHERE skill CONTAINS interest)
              OR toLower(c.title) CONTAINS interest
              OR toLower(c.summary) CONTAINS interest
            )
            RETURN c, d.level as difficulty, o.name as organization, skills
            ORDER BY 
              CASE d.level
                WHEN $primaryDifficulty THEN 0
                ELSE 1
              END,
              c.rating DESC,
              c.students DESC
            LIMIT 20
          `;

          const fallbackResult = await session.run(fallbackQuery, {
            allowedDifficulties: rules.allowedDifficulties,
            fallbackEnrollment: neo4j.int(Math.floor(rules.minEnrollment * percentage)),
            primaryDifficulty: rules.allowedDifficulties[0],
            interests: interestList
          });

          // Filter by duration with relaxed constraint
          let fallbackFilteredRecords = fallbackResult.records;
          if (rules.maxDurationMonths !== null && durationMultiplier !== null) {
            const relaxedMaxDuration = Math.ceil(rules.maxDurationMonths * durationMultiplier);
            fallbackFilteredRecords = fallbackResult.records.filter(record => {
              const course = record.get('c').properties;
              const durationMonths = parseCourseDuration(course.time);
              return durationMonths === null || durationMonths <= relaxedMaxDuration;
            });
          }

          // Return as soon as we find at least 1 result
          if (fallbackFilteredRecords.length >= 1) {
            const recommendations = fallbackFilteredRecords.map(record => {
              const course = record.get('c').properties;
              const skills = record.get('skills');
              const difficulty = record.get('difficulty');
              
              const matchedSkills = interestList.filter(interest => 
                skills.some(skill => skill.includes(interest))
              );

              const matchedInTitle = interestList.filter(interest => 
                course.title.toLowerCase().includes(interest)
              );

              const allMatches = [...new Set([...matchedSkills, ...matchedInTitle])];

              return {
                title: course.title,
                rating: course.rating ? parseFloat(course.rating) : 0,
                reviewCount: course.review_num ? neo4j.int(course.review_num).toNumber() : 0,
                duration: course.time,
                students: course.students ? neo4j.int(course.students).toNumber() : 0,
                url: course.url,
                summary: course.summary,
                difficulty: difficulty,
                organization: record.get('organization'),
                skills: skills,
                reason: allMatches.length > 0 
                  ? `${difficulty} level - Teaches: ${allMatches.slice(0, 3).join(', ')}`
                  : `Recommended ${difficulty} course for your level`
              };
            });

            return res.json({ recommendations: recommendations.slice(0, 10) });
          }
        }
      }

      // If still not found after all fallback attempts, return empty
      return res.json({ recommendations: [] });
    }

    // Return up to 10 recommendations if found (minimum 1)
    const recommendations = filteredRecords.map(record => {
      const course = record.get('c').properties;
      const skills = record.get('skills');
      const difficulty = record.get('difficulty');
      
      const matchedSkills = interestList.filter(interest => 
        skills.some(skill => skill.includes(interest))
      );

      const matchedInTitle = interestList.filter(interest => 
        course.title.toLowerCase().includes(interest)
      );

      const allMatches = [...new Set([...matchedSkills, ...matchedInTitle])];

      // Build reason based on matches and popularity
      let reason = '';
      if (allMatches.length > 0) {
        reason = `${difficulty} - Teaches: ${allMatches.slice(0, 3).join(', ')}`;
      } else {
        reason = `Popular ${difficulty} course`;
      }

      if (course.students >= 50000) {
        reason += ' • Highly popular';
      }

      return {
        title: course.title,
        rating: course.rating ? parseFloat(course.rating) : 0,
        reviewCount: course.review_num ? neo4j.int(course.review_num).toNumber() : 0,
        duration: course.time,
        students: course.students ? neo4j.int(course.students).toNumber() : 0,
        url: course.url,
        summary: course.summary,
        difficulty: difficulty,
        organization: record.get('organization'),
        skills: skills,
        reason: reason
      };
    });

    res.json({ recommendations: recommendations.slice(0, 10) });
  } catch (error) {
    console.error('Personalized recommendations error:', error);
    res.status(500).json({ error: 'Failed to fetch recommendations' });
  } finally {
    await session.close();
  }
});

// Get learning path from course A to course B
app.get('/api/recommendations/learning-path', async (req, res) => {
  const { from, to } = req.query;

  if (!from || !to) {
    return res.status(400).json({ error: 'Both "from" and "to" course titles are required' });
  }

  const session = driver.session();
  try {
    // Find shortest path based on RECOMMENDS_AFTER relationships
    const result = await session.run(
      `MATCH path = shortestPath((start:Course {title: $from})-[:RECOMMENDS_AFTER*1..5]->(end:Course {title: $to}))
       RETURN [node in nodes(path) | node.title] as coursePath,
              [rel in relationships(path) | rel.reason] as reasons,
              length(path) as pathLength`,
      { from: decodeURIComponent(from), to: decodeURIComponent(to) }
    );

    if (result.records.length === 0) {
      return res.status(404).json({ 
        error: 'No learning path found between these courses',
        suggestion: 'Try selecting courses with related skills or difficulty progression'
      });
    }

    const record = result.records[0];
    const path = {
      courses: record.get('coursePath'),
      reasons: record.get('reasons'),
      length: record.get('pathLength').toNumber()
    };

    res.json({ path });
  } catch (error) {
    console.error('Learning path error:', error);
    res.status(500).json({ error: 'Failed to find learning path' });
  } finally {
    await session.close();
  }
});

// Get learning pathway from beginner to intermediate based on topic
app.get('/api/recommendations/next/:courseTitle', async (req, res) => {
  const { courseTitle } = req.params;
  const { userLevel = '1' } = req.query; // Default to level 1
  
  const session = driver.session();

  try {
    // Define level rules - Level 1 most relaxed, Level 5 most strict
    // Level 3-5: NO Beginner courses, only Intermediate+
    const levelRules = {
      '1': { 
        minEnrollment: 500, 
        maxDurationMonths: null,
        minRating: 4.0,
        allowedDifficulties: ['Beginner', 'Intermediate', 'Others']
      }, // Most relaxed
      '2': { 
        minEnrollment: 2000, 
        maxDurationMonths: 10,
        minRating: 4.3,
        allowedDifficulties: ['Beginner', 'Intermediate', 'Others']
      },
      '3': { 
        minEnrollment: 5000, 
        maxDurationMonths: 4,
        minRating: 4.5,
        allowedDifficulties: ['Intermediate', 'Others'] // NO Beginner
      },
      '4': { 
        minEnrollment: 10000, 
        maxDurationMonths: 2,
        minRating: 4.6,
        allowedDifficulties: ['Intermediate', 'Others'] // NO Beginner
      },
      '5': { 
        minEnrollment: 30000, 
        maxDurationMonths: 1,
        minRating: 4.7,
        allowedDifficulties: ['Intermediate', 'Others'] // NO Beginner
      } // Most strict
    };

    const rules = levelRules[userLevel] || levelRules['1'];

    // Search for courses matching the topic/keyword
    const searchTerm = decodeURIComponent(courseTitle).toLowerCase();
    
    const query = `
      MATCH (c:Course)-[:HAS_DIFFICULTY]->(d:Difficulty)
      WHERE d.level IN $allowedDifficulties
      AND c.students >= $minEnrollment
      AND c.rating >= $minRating
      OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
      OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
      WITH c, d, o, collect(DISTINCT toLower(s.name)) as skills
      WHERE toLower(c.title) CONTAINS $searchTerm
      OR toLower(c.summary) CONTAINS $searchTerm
      OR any(skill IN skills WHERE skill CONTAINS $searchTerm)
      RETURN c, d.level as difficulty, o.name as organization, skills
      ORDER BY 
        CASE d.level
          WHEN 'Beginner' THEN 0
          WHEN 'Intermediate' THEN 1
          ELSE 2
        END,
        c.rating DESC,
        c.students DESC
      LIMIT 30
    `;

    const result = await session.run(query, {
      searchTerm: searchTerm.toLowerCase(),
      minEnrollment: neo4j.int(rules.minEnrollment),
      minRating: rules.minRating,
      allowedDifficulties: rules.allowedDifficulties
    });
    
    console.log(`[Learning Path] Initial query results: ${result.records.length}`);

    // Filter by duration if specified
    let filteredRecords = result.records;
    if (rules.maxDurationMonths !== null) {
      filteredRecords = result.records.filter(record => {
        const course = record.get('c').properties;
        const durationMonths = parseCourseDuration(course.time);
        return durationMonths === null || durationMonths <= rules.maxDurationMonths;
      });
      console.log(`[Learning Path] After duration filter (max ${rules.maxDurationMonths} months): ${filteredRecords.length}`);
    }

    // If less than 5 courses, try multiple fallback strategies
    if (filteredRecords.length < 5) {
      const fallbackPercentages = [0.5, 0.3, 0.1, 0.01]; // Enrollment: 50% → 30% → 10% → 1%
      const durationMultipliers = [2, 5, null]; // Duration: 2x → 5x → no limit
      const ratingReductions = [0.2, 0.4, 0.6]; // Rating: -0.2 → -0.4 → -0.6
      
      for (const percentage of fallbackPercentages) {
        for (const durationMultiplier of durationMultipliers) {
          for (const ratingReduction of ratingReductions) {
            const fallbackQuery = `
              MATCH (c:Course)-[:HAS_DIFFICULTY]->(d:Difficulty)
              WHERE d.level IN $allowedDifficulties
              AND c.students >= $fallbackEnrollment
              AND c.rating >= $fallbackRating
              OPTIONAL MATCH (c)-[:OFFERS_SKILL]->(s:Skill)
              OPTIONAL MATCH (c)-[:PROVIDED_BY]->(o:Organization)
              WITH c, d, o, collect(DISTINCT toLower(s.name)) as skills
              WHERE toLower(c.title) CONTAINS $searchTerm
              OR toLower(c.summary) CONTAINS $searchTerm
              OR any(skill IN skills WHERE skill CONTAINS $searchTerm)
              RETURN c, d.level as difficulty, o.name as organization, skills
              ORDER BY 
                c.rating DESC,
                c.students DESC
              LIMIT 30
            `;

            const fallbackResult = await session.run(fallbackQuery, {
              searchTerm: searchTerm.toLowerCase(),
              fallbackEnrollment: neo4j.int(Math.floor(rules.minEnrollment * percentage)),
              fallbackRating: Math.max(3.0, rules.minRating - ratingReduction),
              allowedDifficulties: rules.allowedDifficulties
            });

            // Filter by duration if specified
            let fallbackFiltered = fallbackResult.records;
            if (rules.maxDurationMonths !== null && durationMultiplier !== null) {
              const relaxedMaxDuration = Math.ceil(rules.maxDurationMonths * durationMultiplier);
              fallbackFiltered = fallbackResult.records.filter(record => {
                const course = record.get('c').properties;
                const durationMonths = parseCourseDuration(course.time);
                return durationMonths === null || durationMonths <= relaxedMaxDuration;
              });
            }

            if (fallbackFiltered.length >= 5) {
              filteredRecords = fallbackFiltered;
              break;
            }
          }
          
          if (filteredRecords.length >= 5) {
            break;
          }
        }
        
        if (filteredRecords.length >= 5) {
          break;
        }
      }
    }

    // Build pathway: Start with Beginner courses, then Intermediate
    const beginnerCourses = [];
    const intermediateCourses = [];

    filteredRecords.forEach(record => {
      const course = record.get('c').properties;
      const difficulty = record.get('difficulty');
      const skills = record.get('skills');
      
      const courseData = {
        title: course.title,
        rating: course.rating ? parseFloat(course.rating) : 0,
        reviewCount: course.review_num ? neo4j.int(course.review_num).toNumber() : 0,
        duration: course.time,
        students: course.students ? neo4j.int(course.students).toNumber() : 0,
        url: course.url,
        summary: course.summary,
        difficulty: difficulty,
        organization: record.get('organization'),
        skills: skills,
        reason: difficulty === 'Beginner' 
          ? 'Start here - Foundation course with high enrollment'
          : 'Next step - Build on beginner knowledge'
      };

      if (difficulty === 'Beginner') {
        beginnerCourses.push(courseData);
      } else if (difficulty === 'Intermediate') {
        intermediateCourses.push(courseData);
      }
    });
    
    console.log(`[Learning Path] Beginner: ${beginnerCourses.length}, Intermediate: ${intermediateCourses.length}, UserLevel: ${userLevel}`);

    // Build pathway based on what's available
    // For Level 1-2: Prefer mix of Beginner → Intermediate
    // For Level 3-5: Only Intermediate (Beginner excluded from query)
    let pathway = [];
    const level = parseInt(userLevel);
    
    if (level <= 2) {
      // Level 1-2: Mix Beginner and Intermediate
      pathway = [
        ...beginnerCourses.slice(0, 4),
        ...intermediateCourses.slice(0, 6)
      ];
    } else {
      // Level 3-5: Only Intermediate courses (no Beginner)
      pathway = intermediateCourses.slice(0, 10);
    }

    // Add step numbers and progression reasons
    const recommendations = pathway.slice(0, 10).map((course, index) => ({
      ...course,
      step: index + 1,
      reason: course.difficulty === 'Beginner'
        ? (index === 0 ? '🎯 Start here - Perfect for beginners' : `📚 Step ${index + 1} - Continue building foundations`)
        : `🚀 Step ${index + 1} - ${level >= 3 ? 'Advanced learning path' : 'Ready for intermediate challenges'}`
    }));

    // Minimum threshold depends on level:
    // Level 1-2: Allow 3+ courses (can be all Beginner if no Intermediate)
    // Level 3-5: Need 5+ courses (only Intermediate)
    const minCourses = level <= 2 ? 3 : 5;
    
    if (recommendations.length < minCourses) {
      return res.json({ 
        recommendations: [],
        message: 'Not enough courses found. Try a different topic or lower user level.'
      });
    }

    res.json({ recommendations });
  } catch (error) {
    console.error('Learning pathway error:', error);
    res.status(500).json({ error: 'Failed to fetch learning pathway' });
  } finally {
    await session.close();
  }
});

// ============================================================
// USER PROGRESS ROUTES
// ============================================================

// Mark course as completed (disabled - authentication removed)
app.post('/api/user/complete-course', async (req, res) => {
  res.status(501).json({ error: 'User authentication is disabled. This feature is not available.' });
});

// Add course to wishlist (disabled - authentication removed)
app.post('/api/user/wishlist', async (req, res) => {
  res.status(501).json({ error: 'User authentication is disabled. This feature is not available.' });
});

// Get user's completed courses (disabled - authentication removed)
app.get('/api/user/completed', async (req, res) => {
  res.status(501).json({ error: 'User authentication is disabled. This feature is not available.' });
});

// Get user's wishlist (disabled - authentication removed)
app.get('/api/user/wishlist', async (req, res) => {
  res.status(501).json({ error: 'User authentication is disabled. This feature is not available.' });
});

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
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth/*',
      courses: '/api/courses',
      recommendations: '/api/recommendations/*',
      user: '/api/user/*'
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
  console.log(`  - Auth: http://localhost:${PORT}/api/auth/*`);
  console.log(`  - Courses: http://localhost:${PORT}/api/courses`);
  console.log(`  - Recommendations: http://localhost:${PORT}/api/recommendations/*`);
});
