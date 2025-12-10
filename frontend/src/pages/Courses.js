import React, { useState, useEffect } from 'react';
import axios from 'axios';
import CourseCard from '../components/CourseCard';
import { toast } from 'react-toastify';
import './Pages.css';

const Courses = () => {
  const [courses, setCourses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [difficulty, setDifficulty] = useState('');
  const [page, setPage] = useState(1);

  useEffect(() => {
    fetchCourses();
  }, [page, difficulty]);

  const fetchCourses = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/courses', {
        params: { page, limit: 20, difficulty, search }
      });
      setCourses(response.data.courses || []);
    } catch (error) {
      console.error('Error fetching courses:', error);
      toast.error('Failed to load courses');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    setPage(1);
    fetchCourses();
  };

  if (loading) return <div className="loading-container"><div className="spinner"></div></div>;

  return (
    <div className="page">
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">Browse Courses</h1>
          <p className="page-subtitle">Explore over 1000 courses from top organizations</p>
        </div>

        <div className="card" style={{ marginBottom: '30px' }}>
          <form onSubmit={handleSearch} style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search courses..."
              style={{ 
                flex: 1, 
                minWidth: '200px',
                padding: '10px', 
                border: '1px solid var(--border-color)', 
                borderRadius: '6px',
                backgroundColor: 'var(--bg-secondary)',
                color: 'var(--text-primary)'
              }}
            />
            <select
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              style={{ 
                padding: '10px', 
                border: '1px solid var(--border-color)', 
                borderRadius: '6px',
                backgroundColor: 'var(--bg-secondary)',
                color: 'var(--text-primary)'
              }}
            >
              <option value="">All Levels</option>
              <option value="Beginner">Beginner</option>
              <option value="Intermediate">Intermediate</option>
              <option value="Advanced">Advanced</option>
              <option value="Expert">Expert</option>
            </select>
            <button type="submit" className="btn btn-primary">Search</button>
          </form>
        </div>

        <div className="grid grid-cols-3">
          {courses.length > 0 ? (
            courses.map((course, index) => (
              <CourseCard key={index} course={course} />
            ))
          ) : (
            <div className="empty-state">
              <p className="empty-state-title">No courses found</p>
              <p className="empty-state-text">Try adjusting your search or filters</p>
            </div>
          )}
        </div>

        {courses.length > 0 && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '30px' }}>
            <button 
              onClick={() => setPage(p => Math.max(1, p - 1))} 
              disabled={page === 1}
              className="btn btn-secondary"
            >
              Previous
            </button>
            <span style={{ padding: '10px 20px', color: 'var(--text-secondary)' }}>Page {page}</span>
            <button 
              onClick={() => setPage(p => p + 1)}
              className="btn btn-secondary"
              disabled={courses.length < 20}
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Courses;
