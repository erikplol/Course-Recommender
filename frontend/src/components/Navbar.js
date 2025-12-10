import React from 'react';
import { Link } from 'react-router-dom';
import { FaGraduationCap, FaMoon, FaSun } from 'react-icons/fa';
import './Navbar.css';

const Navbar = ({ darkMode, setDarkMode }) => {

  return (
    <nav className="navbar">
      <div className="container navbar-container">
        <Link to="/" className="navbar-brand">
          <FaGraduationCap className="brand-icon" />
          <span>CourseRecommender</span>
        </Link>

        <div className="navbar-menu">
          <Link to="/courses" className="nav-link">Courses</Link>
          <Link to="/recommendations" className="nav-link">Recommendations</Link>
          <Link to="/learning-path" className="nav-link">Learning Path</Link>
        </div>

        <div className="navbar-actions">
          <button
            className="theme-toggle"
            onClick={() => setDarkMode(!darkMode)}
            aria-label="Toggle dark mode"
          >
            {darkMode ? <FaSun /> : <FaMoon />}
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
