import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { FaStar, FaUsers, FaClock, FaArrowLeft, FaExternalLinkAlt } from 'react-icons/fa';
import './Pages.css';

const CourseDetail = () => {
  const { title } = useParams();
  const [course, setCourse] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCourse();
  }, [title]);

  const fetchCourse = async () => {
    try {
      const response = await axios.get(`/api/courses/${title}`);
      setCourse(response.data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="loading-container"><div className="spinner"></div></div>;
  if (!course) return <div className="error-container"><p>Course not found</p></div>;

  return (
    <div className="page">
      <div className="container">
        <Link to="/courses" className="btn btn-secondary" style={{marginBottom: '20px', display: 'inline-flex'}}>
          <FaArrowLeft /> Back to Courses
        </Link>
        <h1 className="page-title">{course.title}</h1>
        {course.organization && <p className="text-secondary">{course.organization}</p>}
        
        {course.url && (
          <a 
            href={course.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="btn btn-primary"
            style={{marginTop: '15px', display: 'inline-flex', alignItems: 'center', gap: '8px'}}
          >
            View Course on Coursera <FaExternalLinkAlt size={14} />
          </a>
        )}
        
        <div className="course-meta" style={{marginTop: '20px'}}>
          {course.rating && <span><FaStar /> {course.rating}</span>}
          {course.students && <span><FaUsers /> {course.students}</span>}
          {course.duration && <span><FaClock /> {course.duration}</span>}
        </div>
        <p style={{marginTop: '20px'}}>{course.description}</p>
        {course.skills && course.skills.length > 0 && (
          <div style={{marginTop: '20px'}}>
            <h3>Skills</h3>
            <div className="course-skills">
              {course.skills.map((skill, i) => <span key={i} className="skill-tag">{skill}</span>)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CourseDetail;
