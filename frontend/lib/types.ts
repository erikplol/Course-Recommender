export interface Course {
  title: string;
  rating: number;
  reviewCount: number;
  duration: string;
  students: number;
  url: string;
  summary: string;
  description?: string;
  difficulty: string;
  organization: string;
  certificateType?: string;
  skills: string[];
  relatedCourses?: Array<{ title: string; rating: number }>;
  prerequisites?: Array<{ title: string; rating: number }>;
  nextCourses?: Array<{ title: string; rating: number }>;
  reason?: string;
  step?: number;
  score?: number;
  explanation?: string;
  connections?: Array<{
    from: string;
    type: string;
    reason: string;
  }>;
}

export interface RecommendationResponse {
  recommendations: Course[];
  metadata?: {
    totalCandidates: number;
    userSkills: string[];
    learningGoals: string[];
    availableTime: string;
    userLevel: number;
    pathLength: number;
    stages: {
      '1_user_input': string;
      '2_neo4j_query': string;
      '3_rule_engine': string;
      '4_visualization': string;
    };
  };
  message?: string;
}

export interface Organization {
  name: string;
  courseCount: number;
}

export interface Skill {
  name: string;
  courseCount: number;
}
