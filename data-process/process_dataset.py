import pandas as pd
import spacy
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter, defaultdict
import networkx as nx
import json
from typing import List, Dict, Set, Tuple


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    """Centralized configuration for the course processing pipeline."""
    
    # Neo4j Database
    NEO4J_URI = "neo4j+s://f7d62c0a.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASS = "VBRuirRqvWKo94uoeq8vJEveOivYZFpoFsh2KNH-VaY"
    
    # Dataset
    DATASET_PATH = "archive/coursera_courses.csv"
    
    # Skill Extraction
    SKILL_EXTRACTION_MIN_LEN = 4
    SPACY_MODEL = "en_core_web_sm"
    
    # Course Similarity
    RELATED_TO_THRESHOLD = 0.6
    MAX_RELATED_COURSES = 3
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    
    # Prerequisite Discovery
    MIN_COOCCURRENCE = 2
    DIFFICULTY_GAP_THRESHOLD = 0.3
    CONFIDENCE_THRESHOLD = 0.2
    SEMANTIC_SIMILARITY_THRESHOLD = 0.4
    MAX_SKILLS_FOR_ANALYSIS = 200
    
    # Memory optimization for prerequisite inference
    MAX_SKILLS_FOR_PREREQUISITE_INFERENCE = 150
    MIN_SKILL_FREQUENCY = 2
    
    # Difficulty Mapping
    DIFFICULTY_SCORES = {
        "Beginner": 1,
        "Intermediate": 2,
        "Advanced": 3,
        "Expert": 4,
        "Unknown": 2
    }
    DIFFICULTY_ORDER = ["Beginner", "Intermediate", "Advanced", "Expert"]


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================
class DataPreprocessor:
    """Handles dataset loading and preprocessing."""
    
    @staticmethod
    def load_and_preprocess(file_path: str) -> pd.DataFrame:
        """Load and preprocess the course dataset.
        
        Args:
            file_path: Path to the CSV dataset
            
        Returns:
            Preprocessed DataFrame
        """
        print("="*60)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        df = pd.read_csv(file_path)
        print(f"Original dataset size: {len(df)} rows")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['course_title'], keep='first')
        print(f"After removing duplicates: {len(df)} rows")
        
        # Handle null values
        df = df.dropna(subset=['course_title'])
        
        # Normalize categorical data
        df['course_difficulty'] = df['course_difficulty'].fillna('Unknown').str.strip().str.title()
        df['course_certificate_type'] = df['course_certificate_type'].fillna('Unknown').str.strip().str.title()
        df['course_organization'] = df['course_organization'].fillna('Unknown').str.strip()
        
        # Standardize text fields
        text_columns = ['course_summary', 'course_description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').str.lower().str.strip()
        
        # Parse course_skills if it exists
        if 'course_skills' in df.columns:
            df['course_skills'] = df['course_skills'].fillna('').apply(
                lambda x: [s.strip() for s in str(x).split(',') if s.strip()] if x else []
            )
        else:
            df['course_skills'] = [[] for _ in range(len(df))]
        
        # Create combined text field for NLP (including skills for context)
        df["full_text"] = df[
            ["course_title", "course_summary", "course_description"]
        ].fillna("").agg(" ".join, axis=1)
        
        print("✓ Data preprocessing completed\n")
        return df


# ============================================================
# SKILL EXTRACTION
# ============================================================
class SkillExtractor:
    """Extracts skills from course text using NLP."""
    
    def __init__(self, model_name: str = Config.SPACY_MODEL):
        """Initialize the skill extractor with spaCy model."""
        self.nlp = spacy.load(model_name)
        self.min_length = Config.SKILL_EXTRACTION_MIN_LEN
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using NLP.
        
        Args:
            text: Input text to extract skills from
            
        Returns:
            List of extracted skill names
        """
        doc = self.nlp(text)
        skills = set()
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if len(chunk_text) >= self.min_length:
                skills.add(chunk_text.lower())
        
        # Extract important nouns and proper nouns
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) >= self.min_length:
                skills.add(token.text.lower())
        
        return list(skills)
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use dataset skills and supplement with NLP extraction.
        
        Args:
            df: DataFrame with course data
            
        Returns:
            DataFrame with added 'extracted_skills' column
        """
        print("="*60)
        print("STEP 2: PROCESSING SKILLS")
        print("="*60)
        
        def combine_skills(row):
            # Start with dataset skills (primary source)
            dataset_skills = set(row.get('course_skills', []))
            
            # Add NLP-extracted skills from title and summary for enrichment
            if row.get('course_title') or row.get('course_summary'):
                text = f"{row.get('course_title', '')} {row.get('course_summary', '')}"
                nlp_skills = set(self.extract_skills(text))
                # Only add NLP skills that are substantial (avoid noise)
                nlp_skills = {s for s in nlp_skills if len(s.split()) <= 3}  # Max 3 words
                dataset_skills.update(nlp_skills)
            
            return list(dataset_skills)
        
        df["extracted_skills"] = df.apply(combine_skills, axis=1)
        
        dataset_skill_count = sum(len(row.get('course_skills', [])) for _, row in df.iterrows())
        total_skills = len(set(skill for skills in df["extracted_skills"] for skill in skills))
        print(f"✓ Used {dataset_skill_count} skills from dataset")
        print(f"✓ Total unique skills: {total_skills}\n")
        
        return df


# ============================================================
# NEO4J DATABASE MANAGER
# ============================================================
class Neo4jManager:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all existing data from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def create_constraints(self):
        """Create uniqueness constraints for all node types."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Course) REQUIRE c.title IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Difficulty) REQUIRE d.level IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ct:CertificateType) REQUIRE ct.type IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    pass  # Constraint already exists
    
    def initialize_database(self):
        """Initialize database by clearing data and creating constraints."""
        print("="*60)
        print("STEP 3: INITIALIZING NEO4J DATABASE")
        print("="*60)
        
        self.clear_database()
        self.create_constraints()
        
        print("✓ Database cleared and constraints created\n")
    
    def insert_course_with_relationships(self, course_data: Dict, skills: List[str]):
        """Insert a course and all its relationships into Neo4j.
        
        Args:
            course_data: Dictionary containing course information
            skills: List of skills for this course
        """
        with self.driver.session() as session:
            session.execute_write(self._create_course_transaction, course_data, skills)
    
    @staticmethod
    def _create_course_transaction(tx, course_data: Dict, skills: List[str]):
        """Transaction to create course and all related nodes."""
        
        # Create Course node
        tx.run("""
            MERGE (c:Course {title: $title})
            SET c.rating = $rating,
                c.review_num = $review_num,
                c.time = $time,
                c.students = $students,
                c.url = $url,
                c.summary = $summary,
                c.description = $description
        """, course_data)
        
        # Create Organization relationship
        tx.run("""
            MERGE (o:Organization {name: $org_name})
            WITH o
            MATCH (c:Course {title: $title})
            MERGE (c)-[:PROVIDED_BY]->(o)
        """, {"title": course_data["title"], "org_name": course_data["organization"]})
        
        # Create Difficulty relationship
        tx.run("""
            MERGE (d:Difficulty {level: $difficulty})
            WITH d
            MATCH (c:Course {title: $title})
            MERGE (c)-[:HAS_DIFFICULTY]->(d)
        """, {"title": course_data["title"], "difficulty": course_data["difficulty"]})
        
        # Create CertificateType relationship
        tx.run("""
            MERGE (ct:CertificateType {type: $cert_type})
            WITH ct
            MATCH (c:Course {title: $title})
            MERGE (c)-[:HAS_CERTIFICATE_TYPE]->(ct)
        """, {"title": course_data["title"], "cert_type": course_data["cert_type"]})
        
        # Create Skill relationships
        for skill in skills:
            tx.run("""
                MERGE (s:Skill {name: $skill})
                WITH s
                MATCH (c:Course {title: $title})
                MERGE (c)-[:OFFERS_SKILL]->(s)
            """, {"title": course_data["title"], "skill": skill})
    
    def get_all_skills(self) -> Set[str]:
        """Get all skill names from the database.
        
        Returns:
            Set of skill names
        """
        with self.driver.session() as session:
            result = session.run("MATCH (s:Skill) RETURN s.name as name")
            return set(record["name"] for record in result)
    
    def create_skill_prerequisite(self, skill: str, prerequisite: str):
        """Create a prerequisite relationship between skills."""
        with self.driver.session() as session:
            session.execute_write(
                self._create_prerequisite_transaction,
                skill,
                prerequisite
            )
    
    @staticmethod
    def _create_prerequisite_transaction(tx, skill: str, prerequisite: str):
        """Transaction to create a prerequisite relationship."""
        tx.run("""
            MATCH (s1:Skill {name: $prerequisite})
            MATCH (s2:Skill {name: $skill})
            MERGE (s1)-[:PREREQUISITE_FOR]->(s2)
        """, {"skill": skill, "prerequisite": prerequisite})
    
    def create_course_recommendation(self, course_from: str, course_to: str, reason: str):
        """Create a recommendation relationship between courses."""
        with self.driver.session() as session:
            session.execute_write(
                self._create_recommendation_transaction,
                course_from,
                course_to,
                reason
            )
    
    @staticmethod
    def _create_recommendation_transaction(tx, course_from: str, course_to: str, reason: str):
        """Transaction to create a recommendation relationship."""
        tx.run("""
            MATCH (c1:Course {title: $from})
            MATCH (c2:Course {title: $to})
            MERGE (c1)-[r:RECOMMENDS_AFTER]->(c2)
            SET r.reason = $reason
        """, {"from": course_from, "to": course_to, "reason": reason})
    
    def get_skill_based_recommendations(self) -> List[Dict]:
        """Get course recommendations based on skill prerequisites.
        
        Returns:
            List of recommendation dictionaries
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c1:Course)-[:OFFERS_SKILL]->(s1:Skill)
                MATCH (s1)-[:PREREQUISITE_FOR]->(s2:Skill)
                MATCH (c2:Course)-[:OFFERS_SKILL]->(s2)
                WHERE c1 <> c2
                RETURN DISTINCT c1.title as from_course, c2.title as to_course, 
                       s1.name as prereq_skill, s2.name as target_skill
            """)
            return [dict(record) for record in result]
    
    def get_difficulty_based_recommendations(self) -> List[Dict]:
        """Get course recommendations based on difficulty progression.
        
        Returns:
            List of recommendation dictionaries
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c1:Course)-[:RELATED_TO]->(c2:Course)
                MATCH (c1)-[:HAS_DIFFICULTY]->(d1:Difficulty)
                MATCH (c2)-[:HAS_DIFFICULTY]->(d2:Difficulty)
                WHERE d1.level <> d2.level
                RETURN c1.title as course1, c2.title as course2, 
                       d1.level as diff1, d2.level as diff2
            """)
            return [dict(record) for record in result]
    
    def create_related_to_relationship(self, course_a: str, course_b: str, score: float):
        """Create a RELATED_TO relationship between courses."""
        with self.driver.session() as session:
            session.execute_write(
                self._create_related_to_transaction,
                course_a,
                course_b,
                score
            )
    
    @staticmethod
    def _create_related_to_transaction(tx, course_a: str, course_b: str, score: float):
        """Transaction to create a RELATED_TO relationship."""
        tx.run("""
            MATCH (a:Course {title: $a}), (b:Course {title: $b})
            MERGE (a)-[r:RELATED_TO]->(b)
            SET r.similarity = $score
        """, {"a": course_a, "b": course_b, "score": score})


# ============================================================
# COURSE SIMILARITY ANALYZER
# ============================================================
class CourseSimilarityAnalyzer:
    """Analyzes course similarity using embeddings."""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        """Initialize the similarity analyzer."""
        self.embedding_model = SentenceTransformer(model_name)
    
    def compute_similarities(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Compute similarity matrix for all courses.
        
        Args:
            df: DataFrame with course data
            
        Returns:
            Tuple of (similarity_matrix, course_titles)
        """
        print("="*60)
        print("STEP 5: COMPUTING COURSE SIMILARITIES")
        print("="*60)
        
        texts = df["full_text"].fillna("").tolist()
        titles = df["course_title"].tolist()
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        print(f"✓ Computed similarities for {len(titles)} courses\n")
        return similarity_matrix, titles
    
    def create_related_relationships(
        self,
        db: Neo4jManager,
        similarity_matrix: np.ndarray,
        titles: List[str]
    ):
        """Create RELATED_TO relationships based on similarity.
        
        Args:
            db: Neo4j database manager
            similarity_matrix: Course similarity matrix
            titles: List of course titles
        """
        print("="*60)
        print("STEP 6: CREATING SIMILARITY RELATIONSHIPS")
        print("="*60)
        
        total_relationships = 0
        
        for i, course_title in enumerate(titles):
            # Get similarity scores
            sims = list(enumerate(similarity_matrix[i]))
            sims = sorted(sims, key=lambda x: x[1], reverse=True)
            
            count = 0
            for j, sim_score in sims:
                if i == j:  # Skip self
                    continue
                if sim_score < Config.RELATED_TO_THRESHOLD:
                    break
                
                db.create_related_to_relationship(
                    course_title,
                    titles[j],
                    float(sim_score)
                )
                count += 1
                total_relationships += 1
                
                if count >= Config.MAX_RELATED_COURSES:
                    break
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(titles)} courses...")
        
        print(f"✓ Created {total_relationships} RELATED_TO relationships\n")


# ============================================================
# PREREQUISITE DISCOVERY ENGINE
# ============================================================
class PrerequisiteDiscovery:
    """Discovers skill prerequisites using multiple methods."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize prerequisite discovery with course data."""
        self.df = df
        self.skill_course_count = Counter()
        self.skill_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.skill_difficulty_distribution = defaultdict(lambda: defaultdict(int))
        self.skill_difficulty_scores = {}
        self.inferred_prerequisites = defaultdict(list)
        
    def analyze_skill_patterns(self):
        """Analyze skill co-occurrence and difficulty patterns."""
        print("="*60)
        print("STEP 7: AUTOMATED PREREQUISITE DISCOVERY")
        print("="*60)
        print("Analyzing skill patterns...")
        
        for _, row in self.df.iterrows():
            skills = row["extracted_skills"]
            difficulty = row.get("course_difficulty", "Unknown")
            
            # Count skill occurrences
            for skill in skills:
                self.skill_course_count[skill] += 1
                self.skill_difficulty_distribution[skill][difficulty] += 1
            
            # Count co-occurrences
            for i, skill1 in enumerate(skills):
                for skill2 in skills[i+1:]:
                    self.skill_cooccurrence[skill1][skill2] += 1
                    self.skill_cooccurrence[skill2][skill1] += 1
        
        print(f"✓ Analyzed {len(self.skill_course_count)} unique skills")
        
        # Calculate difficulty scores
        self._calculate_difficulty_scores()
    
    def _calculate_difficulty_scores(self):
        """Calculate average difficulty score for each skill."""
        for skill in self.skill_course_count.keys():
            dist = self.skill_difficulty_distribution[skill]
            if not dist:
                self.skill_difficulty_scores[skill] = 2  # Default intermediate
                continue
            
            total = sum(dist.values())
            weighted_sum = sum(
                Config.DIFFICULTY_SCORES.get(diff, 2) * count 
                for diff, count in dist.items()
            )
            self.skill_difficulty_scores[skill] = weighted_sum / total if total > 0 else 2
    
    def infer_from_difficulty_progression(self):
        """Method 1: Infer prerequisites from difficulty progression.
        
        Improved logic:
        - Uses actual skill names to identify foundational vs advanced skills
        - Better filtering for meaningful prerequisites
        - Higher confidence scoring
        """
        print("\nMethod 1: Difficulty Progression & Semantic Analysis")
        
        # Keywords indicating foundational/prerequisite skills
        foundational_keywords = ['introduction', 'basics', 'fundamentals', 'getting started', 
                                'beginner', 'intro', 'basic', 'foundation', 'essentials']
        advanced_keywords = ['advanced', 'expert', 'mastery', 'deep', 'professional', 
                           'specialized', 'applied', 'practical application']
        
        # Get top skills by frequency
        top_skills = [
            skill for skill, count in self.skill_course_count.most_common(
                Config.MAX_SKILLS_FOR_PREREQUISITE_INFERENCE
            )
            if count >= Config.MIN_SKILL_FREQUENCY
        ]
        
        print(f"Analyzing {len(top_skills)} most frequent skills (from {len(self.skill_course_count)} total)")
        
        count = 0
        processed = 0
        
        for skill1 in top_skills:
            # Pre-filter skills that co-occur with skill1
            cooccurring_skills = [
                skill2 for skill2 in top_skills
                if skill1 != skill2 and self.skill_cooccurrence[skill1][skill2] >= Config.MIN_COOCCURRENCE
            ]
            
            diff1 = self.skill_difficulty_scores[skill1]
            skill1_lower = skill1.lower()
            is_foundational = any(kw in skill1_lower for kw in foundational_keywords)
            
            for skill2 in cooccurring_skills:
                diff2 = self.skill_difficulty_scores[skill2]
                skill2_lower = skill2.lower()
                is_advanced = any(kw in skill2_lower for kw in advanced_keywords)
                
                # Better prerequisite detection:
                # 1. Difficulty progression
                # 2. Foundational keywords in prerequisite
                # 3. Advanced keywords in target skill
                # 4. Semantic relationship (e.g., "Python" before "Data Science with Python")
                
                difficulty_match = diff1 < diff2 - Config.DIFFICULTY_GAP_THRESHOLD
                keyword_match = is_foundational or is_advanced
                semantic_match = skill1_lower in skill2_lower and skill1 != skill2
                
                if difficulty_match or keyword_match or semantic_match:
                    cooccur_count = self.skill_cooccurrence[skill1][skill2]
                    base_confidence = cooccur_count / self.skill_course_count[skill2]
                    
                    # Boost confidence for strong signals
                    confidence_boost = 0
                    if semantic_match:
                        confidence_boost += 0.3
                    if is_foundational:
                        confidence_boost += 0.15
                    if difficulty_match:
                        confidence_boost += 0.1
                    
                    final_confidence = min(base_confidence + confidence_boost, 1.0)
                    
                    self.inferred_prerequisites[skill2].append({
                        'prerequisite': skill1,
                        'confidence': final_confidence,
                        'difficulty_gap': diff2 - diff1,
                        'reason': 'difficulty_progression' if difficulty_match else 'semantic_relationship'
                    })
                    count += 1
            
            processed += 1
            if processed % 20 == 0:
                print(f"  Progress: {processed}/{len(top_skills)} skills - Found {count} relationships")
        
        print(f"✓ Found {count} prerequisite relationships")
    
    def refine_with_semantic_similarity(
        self,
        embedding_model: SentenceTransformer
    ) -> Dict[str, List[str]]:
        """Method 3: Refine prerequisites using semantic similarity.
        
        Args:
            embedding_model: Sentence transformer model
            
        Returns:
            Dictionary of refined prerequisites
        """
        print("\nMethod 3: Semantic Similarity Validation")
        
        # Only analyze skills that have inferred prerequisites to save memory
        skills_with_prereqs = list(self.inferred_prerequisites.keys())
        all_relevant_skills = set(skills_with_prereqs)
        
        # Add all prerequisite skills to the set
        for skill_prereqs in self.inferred_prerequisites.values():
            for prereq_info in skill_prereqs:
                all_relevant_skills.add(prereq_info['prerequisite'])
        
        # Limit to most common skills if still too many
        skill_list = [
            skill for skill, _ in self.skill_course_count.most_common(Config.MAX_SKILLS_FOR_ANALYSIS)
            if skill in all_relevant_skills
        ]
        
        print(f"Computing embeddings for {len(skill_list)} relevant skills (reduced from {len(all_relevant_skills)})")
        
        # Generate embeddings in batches to reduce memory
        skill_embeddings = embedding_model.encode(
            skill_list, 
            show_progress_bar=True,
            batch_size=32  # Process in smaller batches
        )
        
        # Compute similarity matrix
        print("Computing similarity matrix...")
        skill_similarity_matrix = cosine_similarity(skill_embeddings)
        
        # Create skill index lookup for O(1) access instead of O(n) list.index()
        skill_to_idx = {skill: idx for idx, skill in enumerate(skill_list)}
        
        # Refine prerequisites based on semantic similarity
        refined_prerequisites = {}
        processed_skills = 0
        
        for skill in self.inferred_prerequisites:
            if skill not in skill_to_idx:
                continue
            
            skill_idx = skill_to_idx[skill]
            refined_prerequisites[skill] = []
            
            for prereq_info in self.inferred_prerequisites[skill]:
                prereq = prereq_info['prerequisite']
                if prereq not in skill_to_idx:
                    continue
                
                prereq_idx = skill_to_idx[prereq]
                semantic_sim = skill_similarity_matrix[skill_idx][prereq_idx]
                
                # Keep if:
                # 1. Very high confidence (>0.6)
                # 2. High confidence (>0.4) with semantic similarity
                # 3. Moderate confidence (>0.2) with high semantic similarity
                if (prereq_info['confidence'] > 0.6 or 
                    (prereq_info['confidence'] > 0.4 and semantic_sim > Config.SEMANTIC_SIMILARITY_THRESHOLD) or
                    (prereq_info['confidence'] > Config.CONFIDENCE_THRESHOLD and semantic_sim > 0.6)):
                    refined_prerequisites[skill].append(prereq)
            
            processed_skills += 1
            if processed_skills % 50 == 0:
                print(f"  Validated {processed_skills}/{len(self.inferred_prerequisites)} skills")
        
        total_refined = sum(len(v) for v in refined_prerequisites.values())
        print(f"✓ Refined to {total_refined} high-confidence prerequisites\n")
        
        # Clear large matrices to free memory
        del skill_similarity_matrix
        del skill_embeddings
        
        return refined_prerequisites
    
    def save_analysis_report(self, refined_prerequisites: Dict[str, List[str]]):
        """Save prerequisite analysis to JSON file.
        
        Args:
            refined_prerequisites: Dictionary of refined prerequisites
        """
        report = {
            "total_skills_analyzed": len(self.skill_course_count),
            "prerequisites_created": sum(len(v) for v in refined_prerequisites.values()),
            "methods_used": ["difficulty_progression", "co_occurrence", "semantic_similarity"],
            "top_skills_with_prerequisites": []
        }
        
        # Add top 20 skills with most prerequisites
        for skill in sorted(
            refined_prerequisites.keys(),
            key=lambda s: len(refined_prerequisites[s]),
            reverse=True
        )[:20]:
            report["top_skills_with_prerequisites"].append({
                "skill": skill,
                "prerequisites": refined_prerequisites[skill],
                "course_count": self.skill_course_count[skill],
                "avg_difficulty": round(self.skill_difficulty_scores[skill], 2)
            })
        
        with open("prerequisite_analysis.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("✓ Analysis report saved to prerequisite_analysis.json")


# ============================================================
# COURSE GRAPH BUILDER
# ============================================================
class CourseGraphBuilder:
    """Builds course and skill relationships in Neo4j."""
    
    def __init__(self, db: Neo4jManager):
        """Initialize graph builder with database connection."""
        self.db = db
    
    def insert_all_courses(self, df: pd.DataFrame):
        """Insert all courses and their relationships into Neo4j.
        
        Args:
            df: DataFrame with course data
        """
        print("="*60)
        print("STEP 4: BUILDING KNOWLEDGE GRAPH")
        print("="*60)
        
        for idx, row in df.iterrows():
            # Convert enrollment to integer (remove commas)
            students_str = str(row.get("course_students_enrolled", "0"))
            students = int(students_str.replace(",", "")) if students_str and students_str != "nan" else 0
            
            course_data = {
                "title": row["course_title"],
                "rating": row.get("course_rating"),
                "review_num": row.get("course_reviews_num"),
                "time": row.get("course_time"),
                "students": students,
                "url": row.get("course_url"),
                "summary": row.get("course_summary"),
                "description": row.get("course_description"),
                "organization": row.get("course_organization", "Unknown"),
                "difficulty": row.get("course_difficulty", "Unknown"),
                "cert_type": row.get("course_certificate_type", "Unknown")
            }
            
            skills = row["extracted_skills"]
            self.db.insert_course_with_relationships(course_data, skills)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} courses...")
        
        print(f"✓ Inserted {len(df)} courses with relationships\n")
    
    def create_prerequisite_relationships(self, refined_prerequisites: Dict[str, List[str]]):
        """Create prerequisite relationships in Neo4j.
        
        Args:
            refined_prerequisites: Dictionary of skill prerequisites
        """
        print("="*60)
        print("STEP 8: CREATING PREREQUISITE RELATIONSHIPS")
        print("="*60)
        
        existing_skills = self.db.get_all_skills()
        prereq_count = 0
        
        for skill, prerequisites in refined_prerequisites.items():
            if skill in existing_skills:
                for prereq in prerequisites:
                    if prereq in existing_skills:
                        self.db.create_skill_prerequisite(skill, prereq)
                        prereq_count += 1
        
        print(f"✓ Created {prereq_count} PREREQUISITE_FOR relationships\n")
    
    def create_course_recommendations(self):
        """Create RECOMMENDS_AFTER relationships with improved strategy.
        
        Strategy:
        1. Skill prerequisites: Course A teaches prerequisite skills for Course B
        2. Difficulty progression: Same organization, increasing difficulty
        3. Topic continuation: Related courses with skill overlap
        
        Returns:
            Total number of recommendation relationships created
        """
        print("="*60)
        print("STEP 9: CREATING COURSE RECOMMENDATIONS")
        print("="*60)
        
        # Strategy 1: Skill-based prerequisites
        print("Creating skill prerequisite recommendations...")
        skill_recommendations = self.db.get_skill_based_recommendations()
        
        # Deduplicate and score skill recommendations
        recommendation_map = {}
        for rec in skill_recommendations:
            key = (rec['from_course'], rec['to_course'])
            if key not in recommendation_map:
                recommendation_map[key] = {
                    'from': rec['from_course'],
                    'to': rec['to_course'],
                    'skills': [],
                    'score': 0
                }
            recommendation_map[key]['skills'].append(
                f"{rec['prereq_skill']} → {rec['target_skill']}"
            )
            recommendation_map[key]['score'] += 1
        
        # Create recommendations with quality threshold
        skill_count = 0
        for rec_data in recommendation_map.values():
            if rec_data['score'] >= 1:  # At least 1 skill prerequisite
                reason = f"Builds on skills: {', '.join(rec_data['skills'][:3])}"
                if len(rec_data['skills']) > 3:
                    reason += f" (+{len(rec_data['skills']) - 3} more)"
                self.db.create_course_recommendation(
                    rec_data['from'],
                    rec_data['to'],
                    reason
                )
                skill_count += 1
        
        print(f"✓ Created {skill_count} skill-based recommendations")
        
        # Strategy 2: Difficulty progression within same organization
        print("Creating difficulty progression recommendations...")
        difficulty_recommendations = self.db.get_difficulty_based_recommendations()
        
        diff_count = 0
        for rec in difficulty_recommendations:
            diff1 = rec['diff1']
            diff2 = rec['diff2']
            
            if (diff1 in Config.DIFFICULTY_ORDER and 
                diff2 in Config.DIFFICULTY_ORDER):
                diff1_idx = Config.DIFFICULTY_ORDER.index(diff1)
                diff2_idx = Config.DIFFICULTY_ORDER.index(diff2)
                
                # Only recommend if exactly one level up
                if diff2_idx == diff1_idx + 1:
                    reason = f"Natural progression: {diff1} → {diff2}"
                    self.db.create_course_recommendation(
                        rec['course1'],
                        rec['course2'],
                        reason
                    )
                    diff_count += 1
        
        print(f"✓ Created {diff_count} difficulty-based recommendations")
        
        # Strategy 3: Topic continuation (related courses with overlapping skills)
        print("Creating topic continuation recommendations...")
        topic_count = self._create_topic_recommendations()
        print(f"✓ Created {topic_count} topic-based recommendations\n")
        
        return skill_count + diff_count + topic_count
    
    def _create_topic_recommendations(self) -> int:
        """Create recommendations based on topic continuation."""
        with self.db.driver.session() as session:
            # Find related courses with significant skill overlap
            result = session.run("""
                MATCH (c1:Course)-[:RELATED_TO]->(c2:Course)
                MATCH (c1)-[:OFFERS_SKILL]->(s:Skill)<-[:OFFERS_SKILL]-(c2)
                MATCH (c1)-[:HAS_DIFFICULTY]->(d1:Difficulty)
                MATCH (c2)-[:HAS_DIFFICULTY]->(d2:Difficulty)
                WHERE d1.level <> d2.level
                WITH c1, c2, d1, d2, count(s) as shared_skills
                WHERE shared_skills >= 2
                RETURN c1.title as course1, c2.title as course2, 
                       d1.level as diff1, d2.level as diff2,
                       shared_skills
                ORDER BY shared_skills DESC
                LIMIT 500
            """)
            
            count = 0
            for rec in result:
                diff1 = rec['diff1']
                diff2 = rec['diff2']
                
                if (diff1 in Config.DIFFICULTY_ORDER and 
                    diff2 in Config.DIFFICULTY_ORDER):
                    if Config.DIFFICULTY_ORDER.index(diff1) < Config.DIFFICULTY_ORDER.index(diff2):
                        reason = f"Continues topic with {rec['shared_skills']} related skills"
                        self.db.create_course_recommendation(
                            rec['course1'],
                            rec['course2'],
                            reason
                        )
                        count += 1
            
            return count


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """Execute the complete course processing pipeline."""
    
    print("\n" + "="*60)
    print("COURSE RECOMMENDATION KNOWLEDGE GRAPH BUILDER")
    print("="*60 + "\n")
    
    # Initialize components
    db = Neo4jManager(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASS)
    preprocessor = DataPreprocessor()
    skill_extractor = SkillExtractor()
    similarity_analyzer = CourseSimilarityAnalyzer()
    
    try:
        # Step 1: Load and preprocess data
        df = preprocessor.load_and_preprocess(Config.DATASET_PATH)
        
        # Step 2: Extract skills
        df = skill_extractor.process_dataframe(df)
        
        # # Step 3: Initialize database
        db.initialize_database()
        
        # # Step 4: Build knowledge graph
        graph_builder = CourseGraphBuilder(db)
        graph_builder.insert_all_courses(df)
        
        # Step 5-6: Compute and create similarity relationships
        similarity_matrix, titles = similarity_analyzer.compute_similarities(df)
        similarity_analyzer.create_related_relationships(db, similarity_matrix, titles)
        
        # Step 7: Discover prerequisites
        prereq_discovery = PrerequisiteDiscovery(df)
        prereq_discovery.analyze_skill_patterns()
        prereq_discovery.infer_from_difficulty_progression()
        refined_prerequisites = prereq_discovery.refine_with_semantic_similarity(
            similarity_analyzer.embedding_model
        )
        prereq_discovery.save_analysis_report(refined_prerequisites)
        
        # Step 8: Create prerequisite relationships
        graph_builder.create_prerequisite_relationships(refined_prerequisites)
        
        # Step 9: Create course recommendations
        total_recommendations = graph_builder.create_course_recommendations()
        
        # Final summary
        print("="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"✓ Courses processed: {len(df)}")
        print(f"✓ Unique skills: {len(prereq_discovery.skill_course_count)}")
        print(f"✓ Prerequisites discovered: {sum(len(v) for v in refined_prerequisites.values())}")
        print(f"✓ Course recommendations: {total_recommendations}")
        print("\n✓ Nodes: Course, Skill, Organization, Difficulty, CertificateType")
        print("✓ Relationships:")
        print("  - OFFERS_SKILL (Course → Skill)")
        print("  - PROVIDED_BY (Course → Organization)")
        print("  - HAS_DIFFICULTY (Course → Difficulty)")
        print("  - HAS_CERTIFICATE_TYPE (Course → CertificateType)")
        print("  - RELATED_TO (Course → Course)")
        print("  - PREREQUISITE_FOR (Skill → Skill)")
        print("  - RECOMMENDS_AFTER (Course → Course)")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    
    finally:
        # Always close database connection
        db.close()
        print("✓ Database connection closed")


if __name__ == "__main__":
    main()