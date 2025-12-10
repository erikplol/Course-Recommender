import React from 'react';
import { Link } from 'react-router-dom';
import { FaStar, FaUsers, FaClock, FaHeart, FaBookmark, FaExternalLinkAlt } from 'react-icons/fa';
import './CourseCard.css';

const CourseCard = ({ course, showActions = false, onWishlist, onComplete }) => {
  const getDifficultyBadgeClass = (difficulty) => {
    const level = difficulty?.toLowerCase();
    if (level === 'beginner') return 'badge-beginner';
    if (level === 'intermediate') return 'badge-intermediate';
    if (level === 'advanced') return 'badge-advanced';
    if (level === 'expert') return 'badge-expert';
    return 'badge-beginner';
  };

  return (
    <div className="course-card card">
      <div className="course-card-header">
        <h3 className="course-title">
          <Link to={`/courses/${encodeURIComponent(course.title)}`}>
            {course.title}
          </Link>
        </h3>
        {course.difficulty && (
          <span className={`badge ${getDifficultyBadgeClass(course.difficulty)}`}>
            {course.difficulty}
          </span>
        )}
      </div>

      {course.organization && (
        <p className="course-organization">{course.organization}</p>
      )}

      {course.summary && (
        <p className="course-summary">{course.summary.substring(0, 120)}...</p>
      )}

      {course.reason && (
        <div className="course-reason">
          <span className="reason-label">Why:</span>
          <span className="reason-text">{course.reason}</span>
        </div>
      )}

      <div className="course-meta">
        {course.rating && (
          <span className="meta-item">
            <FaStar className="meta-icon" />
            {course.rating} {course.reviewCount && `(${course.reviewCount})`}
          </span>
        )}
        {course.students && (
          <span className="meta-item">
            <FaUsers className="meta-icon" />
            {course.students}
          </span>
        )}
        {course.duration && (
          <span className="meta-item">
            <FaClock className="meta-icon" />
            {course.duration}
          </span>
        )}
      </div>

      {course.skills && course.skills.length > 0 && (
        <div className="course-skills">
          {course.skills.slice(0, 3).map((skill, index) => (
            <span key={index} className="skill-tag">{skill}</span>
          ))}
          {course.skills.length > 3 && (
            <span className="skill-tag">+{course.skills.length - 3} more</span>
          )}
        </div>
      )}

      <div className="course-card-footer">
        {course.url && (
          <a 
            href={course.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="btn btn-primary btn-sm"
            style={{display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', padding: '6px 12px'}}
            onClick={(e) => e.stopPropagation()}
          >
            View Course <FaExternalLinkAlt size={12} />
          </a>
        )}
        
        {showActions && (
          <div className="course-actions">
            <button onClick={() => onWishlist?.(course)} className="btn-icon" title="Add to wishlist">
              <FaHeart />
            </button>
            <button onClick={() => onComplete?.(course)} className="btn-icon" title="Mark as completed">
              <FaBookmark />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default CourseCard;
