import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { FaRoute, FaSearch, FaArrowRight } from 'react-icons/fa';
import { Link } from 'react-router-dom';
import './Pages.css';

const LearningPath = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [userLevel, setUserLevel] = useState('1');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchLearningPath = async () => {
    if (!searchTerm.trim()) {
      toast.error('Please enter a topic to search for');
      return;
    }

    setLoading(true);
    setRecommendations([]);
    try {
      const response = await axios.get(
        `/api/recommendations/next/${encodeURIComponent(searchTerm)}`,
        { params: { userLevel } }
      );
      
      if (response.data.recommendations && response.data.recommendations.length > 0) {
        setRecommendations(response.data.recommendations);
        toast.success(`Found ${response.data.recommendations.length} courses in your learning pathway!`);
      } else {
        toast.info(response.data.message || 'No courses found for this topic. Try different keywords.');
        setRecommendations([]);
      }
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to fetch learning pathway');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    fetchLearningPath();
  };

  return (
    <div className="page">
      <div className="container">
        <div className="page-header">
          <h1 className="page-title"><FaRoute /> Learning Pathway Builder</h1>
          <p className="page-subtitle">Get a structured path from beginner to intermediate level (5-10 courses)</p>
        </div>

        <div className="card" style={{marginBottom: '30px'}}>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>What do you want to learn?</label>
              <div style={{position: 'relative'}}>
                <input 
                  type="text" 
                  value={searchTerm} 
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="e.g., Python, Web Development, Data Science, JavaScript..."
                  style={{paddingLeft: '35px'}}
                  required
                />
                <FaSearch style={{position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)'}} />
              </div>
              <small style={{color: 'var(--text-secondary)', fontSize: '0.85rem', display: 'block', marginTop: '5px'}}>
                💡 Enter any topic - doesn't need to be a specific course name
              </small>
            </div>

            <div className="form-group">
              <label>Your Current Level:</label>
              <select 
                value={userLevel} 
                onChange={(e) => setUserLevel(e.target.value)}
                style={{width: '100%'}}
              >
                <option value="1">Level 1 - Complete Beginner</option>
                <option value="2">Level 2 - Beginner (1-2 courses completed)</option>
                <option value="3">Level 3 - Intermediate (Strong basics)</option>
                <option value="4">Level 4 - Upper Intermediate (Regular learner)</option>
                <option value="5">Level 5 - Advanced (Experienced)</option>
              </select>
            </div>

            <button 
              type="submit" 
              className="btn-primary" 
              disabled={loading || !searchTerm.trim()}
              style={{width: '100%'}}
            >
              {loading ? 'Building Pathway...' : '🚀 Generate Learning Pathway'}
            </button>
          </form>
        </div>

        {loading && (
          <div className="loading-container" style={{minHeight: '200px'}}>
            <div className="spinner"></div>
          </div>
        )}

        {!loading && recommendations.length > 0 && (
          <div className="card">
            <h2 style={{marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px'}}>
              <FaRoute style={{color: 'var(--primary-color)'}} />
              Your Learning Pathway ({recommendations.length} Courses)
            </h2>
            <p style={{color: 'var(--text-secondary)', marginBottom: '25px', fontSize: '0.95rem'}}>
              📚 Follow these courses in order from beginner to intermediate
            </p>
            
            <div style={{display: 'flex', flexDirection: 'column', gap: '15px'}}>
              {recommendations.map((course, index) => (
                <div key={index}>
                  <div className="course-card" style={{position: 'relative'}}>
                    <div style={{
                      position: 'absolute',
                      top: '15px',
                      left: '15px',
                      width: '45px',
                      height: '45px',
                      borderRadius: '50%',
                      backgroundColor: course.difficulty === 'Beginner' ? '#10b981' : '#3b82f6',
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontWeight: 'bold',
                      fontSize: '1.2rem',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                    }}>
                      {course.step || index + 1}
                    </div>

                    <div style={{paddingLeft: '70px'}}>
                      <Link 
                        to={`/courses/${encodeURIComponent(course.title)}`}
                        style={{textDecoration: 'none', color: 'inherit'}}
                      >
                        <h3 className="course-title" style={{marginBottom: '8px', fontSize: '1.15rem'}}>
                          {course.title}
                        </h3>
                      </Link>
                      
                      {course.reason && (
                        <p style={{
                          color: course.difficulty === 'Beginner' ? '#10b981' : '#3b82f6',
                          fontSize: '0.9rem',
                          marginBottom: '10px',
                          fontWeight: '500'
                        }}>
                          {course.reason}
                        </p>
                      )}
                      
                      <p className="course-summary" style={{marginBottom: '12px'}}>{course.summary}</p>
                      
                      <div style={{
                        display: 'flex',
                        gap: '15px',
                        marginTop: '12px',
                        flexWrap: 'wrap',
                        alignItems: 'center'
                      }}>
                        {course.difficulty && (
                          <span style={{
                            padding: '5px 14px',
                            backgroundColor: course.difficulty === 'Beginner' ? '#d1fae5' : '#dbeafe',
                            color: course.difficulty === 'Beginner' ? '#10b981' : '#3b82f6',
                            borderRadius: '6px',
                            fontSize: '0.85rem',
                            fontWeight: '600'
                          }}>
                            {course.difficulty}
                          </span>
                        )}
                        {course.rating && (
                          <span style={{fontSize: '0.9rem', color: 'var(--text-secondary)'}}>
                            ⭐ {course.rating}
                          </span>
                        )}
                        {course.students && (
                          <span style={{fontSize: '0.9rem', color: 'var(--text-secondary)'}}>
                            👥 {course.students.toLocaleString()} students
                          </span>
                        )}
                        {course.duration && (
                          <span style={{fontSize: '0.9rem', color: 'var(--text-secondary)'}}>
                            ⏱️ {course.duration}
                          </span>
                        )}
                      </div>

                      {course.url && (
                        <div className="course-card-footer" style={{marginTop: '15px'}}>
                          <a 
                            href={course.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="btn-secondary"
                            style={{fontSize: '0.9rem', display: 'inline-block'}}
                          >
                            View Course on Coursera →
                          </a>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {index < recommendations.length - 1 && (
                    <div style={{
                      display: 'flex',
                      justifyContent: 'center',
                      padding: '8px 0'
                    }}>
                      <FaArrowRight style={{
                        fontSize: '2rem',
                        color: 'var(--primary-color)',
                        transform: 'rotate(90deg)',
                        opacity: 0.5
                      }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && searchTerm && recommendations.length === 0 && (
          <div className="card" style={{textAlign: 'center', padding: '40px 20px'}}>
            <p style={{fontSize: '1.1rem', color: 'var(--text-secondary)', marginBottom: '10px'}}>
              😔 No learning pathway found for "{searchTerm}"
            </p>
            <p style={{fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '15px'}}>
              Try different keywords, broader topics, or a lower user level
            </p>
            <div style={{display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap'}}>
              <Link to="/courses" className="btn-secondary">Browse All Courses</Link>
              <Link to="/recommendations" className="btn-primary">Get Recommendations</Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LearningPath;
