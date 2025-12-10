import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Course API
export const getCourses = async (params?: {
  page?: number;
  limit?: number;
  difficulty?: string;
  organization?: string;
  search?: string;
}) => {
  const response = await api.get('/api/courses', { params });
  return response.data;
};

export const getCourseByTitle = async (title: string) => {
  const response = await api.get(`/api/courses/${encodeURIComponent(title)}`);
  return response.data;
};

// Recommendations API
export const getPersonalizedRecommendations = async (params: {
  skillLevel?: string;
  interests: string;
}) => {
  const response = await api.get('/api/recommendations/personalized', { params });
  return response.data;
};

export const getLearningPath = async (params: {
  from: string;
  to: string;
}) => {
  const response = await api.get('/api/recommendations/learning-path', { params });
  return response.data;
};

export const getNextCourses = async (
  courseTitle: string,
  params?: {
    userLevel?: string;
    currentSkills?: string;
    learningGoals?: string;
    availableTime?: string;
  }
) => {
  const response = await api.get(
    `/api/recommendations/next/${encodeURIComponent(courseTitle)}`,
    { params }
  );
  return response.data;
};

// Filter API
export const getOrganizations = async () => {
  const response = await api.get('/api/organizations');
  return response.data;
};

export const getSkills = async () => {
  const response = await api.get('/api/skills');
  return response.data;
};

export default api;
