import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Navigate } from 'react-router-dom';
import axios from 'axios';
import { FaUser, FaBook, FaHeart } from 'react-icons/fa';
import './Pages.css';

const Profile = () => {
  const { user, isAuthenticated } = useAuth();
  const [completedCourses, setCompletedCourses] = useState([]);
  const [wishlist, setWishlist] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isAuthenticated) {
      fetchUserData();
    }
  }, [isAuthenticated]);

  const fetchUserData = async () => {
    try {
      const [completed, wish] = await Promise.all([
        axios.get('/api/user/completed'),
        axios.get('/api/user/wishlist')
      ]);
      setCompletedCourses(completed.data.courses || []);
      setWishlist(wish.data.courses || []);
    } catch (error) {
      console.error('Error fetching user data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!isAuthenticated) return <Navigate to="/login" />;

  return (
    <div className="page">
      <div className="container">
        <div className="page-header">
          <h1 className="page-title"><FaUser /> My Profile</h1>
        </div>
        
        <div className="card" style={{marginBottom: '30px'}}>
          <h3>Profile Information</h3>
          <p><strong>Username:</strong> {user?.username}</p>
          <p><strong>Email:</strong> {user?.email}</p>
          <p><strong>Skill Level:</strong> {user?.skillLevel}</p>
          <p><strong>Interests:</strong> {user?.interests?.join(', ') || 'None set'}</p>
        </div>

        <div className="grid grid-cols-2" style={{gap: '30px'}}>
          <div className="card">
            <h3><FaBook /> Completed Courses ({completedCourses.length})</h3>
            {loading ? (
              <p>Loading...</p>
            ) : completedCourses.length > 0 ? (
              <ul style={{marginTop: '15px'}}>
                {completedCourses.map((course, i) => (
                  <li key={i} style={{marginBottom: '10px'}}>
                    {course.title}
                    <span className={`badge badge-${course.difficulty?.toLowerCase() || 'beginner'}`} style={{marginLeft: '10px'}}>
                      {course.difficulty}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{marginTop: '10px', color: 'var(--text-secondary)'}}>No completed courses yet</p>
            )}
          </div>

          <div className="card">
            <h3><FaHeart /> Wishlist ({wishlist.length})</h3>
            {loading ? (
              <p>Loading...</p>
            ) : wishlist.length > 0 ? (
              <ul style={{marginTop: '15px'}}>
                {wishlist.map((course, i) => (
                  <li key={i} style={{marginBottom: '10px'}}>
                    {course.title}
                    <span className={`badge badge-${course.difficulty?.toLowerCase() || 'beginner'}`} style={{marginLeft: '10px'}}>
                      {course.difficulty}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{marginTop: '10px', color: 'var(--text-secondary)'}}>No courses in wishlist</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
