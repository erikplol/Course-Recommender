import React, { useState, useEffect } from 'react';
import axios from 'axios';
import CourseCard from '../components/CourseCard';
import { toast } from 'react-toastify';
import { FaLightbulb, FaSearch } from 'react-icons/fa';
import './Pages.css';

const Recommendations = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [skillLevel, setSkillLevel] = useState('');
  const [interests, setInterests] = useState('');
  const [hasSearched, setHasSearched] = useState(false);

  const handleGetRecommendations = async (e) => {
    e.preventDefault();
    
    if (!interests.trim()) {
      toast.error('Please enter your learning interests');
      return;
    }

    setLoading(true);
    setHasSearched(true);
    
    try {
      const params = { interests };
      if (skillLevel) {
        params.skillLevel = skillLevel;
      }
      
      const response = await axios.get('/api/recommendations/personalized', {
        params
      });
      setRecommendations(response.data.recommendations || []);
      
      if (response.data.recommendations?.length > 0) {
        toast.success(`Found ${response.data.recommendations.length} personalized recommendations!`);
      } else {
        toast.info('No recommendations found. Try different interests.');
      }
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to load recommendations');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="container">
        <div className="page-header">
          <h1 className="page-title"><FaLightbulb /> Personalized Recommendations</h1>
          <p className="page-subtitle">Tell us about your skill level and interests to get tailored course recommendations</p>
        </div>

        <div className="card" style={{marginBottom: '30px'}}>
          <form onSubmit={handleGetRecommendations}>
            <div className="form-group">
              <label>Topics You Want to Learn:</label>
              <div style={{position: 'relative'}}>
                <input
                  type="text"
                  value={interests}
                  onChange={(e) => setInterests(e.target.value)}
                  placeholder="e.g., Python, Data Science, Machine Learning, Business Analytics..."
                  required
                  style={{
                    width: '100%',
                    padding: '10px 10px 10px 40px',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    backgroundColor: 'var(--bg-secondary)',
                    color: 'var(--text-primary)',
                    fontSize: '1rem'
                  }}
                />
                <FaSearch style={{
                  position: 'absolute',
                  left: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'var(--text-secondary)'
                }} />
              </div>
            </div>

            <div className="form-group">
              <label>Your Learning Level:</label>
              <select
                value={skillLevel}
                onChange={(e) => setSkillLevel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  backgroundColor: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '1rem'
                }}
              >
                <option value="">All levels (any difficulty)</option>
                <option value="1">Level 1 - Very Beginner (Just starting, new to online learning)</option>
                <option value="2">Level 2 - Beginner (Completed 1-2 courses)</option>
                <option value="3">Level 3 - Intermediate (Have strong basics)</option>
                <option value="4">Level 4 - Upper Intermediate (Learning regularly, want challenges)</option>
                <option value="5">Level 5 - Advanced/Expert (Experienced, took many courses)</option>
              </select>
            </div>

            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={loading}
              style={{width: '100%'}}
            >
              {loading ? 'Finding Recommendations...' : 'Get Personalized Recommendations'}
            </button>
          </form>
        </div>

        {loading && (
          <div className="loading-container" style={{minHeight: '200px'}}>
            <div className="spinner"></div>
          </div>
        )}

        {!loading && hasSearched && recommendations.length > 0 && (
          <div>
            <h2 style={{marginBottom: '20px', fontSize: '1.5rem'}}>
              Your Personalized Recommendations ({recommendations.length})
            </h2>
            <div className="grid grid-cols-3">
              {recommendations.map((course, index) => (
                <CourseCard key={index} course={course} />
              ))}
            </div>
          </div>
        )}

        {!loading && hasSearched && recommendations.length === 0 && (
          <div className="card" style={{textAlign: 'center', padding: '40px 20px'}}>
            <p style={{fontSize: '1.1rem', color: 'var(--text-secondary)', marginBottom: '10px'}}>
              No recommendations found for your criteria.
            </p>
            <p style={{fontSize: '0.9rem', color: 'var(--text-secondary)'}}>
              Try adjusting your skill level or interests, or browse all courses.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Recommendations;
