import React from 'react';
import { Link } from 'react-router-dom';
import { FaRocket, FaChartLine, FaLightbulb, FaGraduationCap } from 'react-icons/fa';
import './Home.css';

const Home = () => {

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <div className="hero-content">
            <h1 className="hero-title">
              Discover Your Perfect <span className="highlight">Learning Path</span>
            </h1>
            <p className="hero-subtitle">
              AI-powered course recommendations using knowledge graph technology.
              Find courses tailored to your skills, interests, and career goals.
            </p>
            <div className="hero-actions">
              <Link to="/recommendations" className="btn btn-primary btn-large">
                <FaRocket />
                Get Recommendations
              </Link>
              <Link to="/courses" className="btn btn-secondary btn-large">
                Browse Courses
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <div className="container">
          <h2 className="section-title">Why Choose CourseRecommender?</h2>
          <div className="grid grid-cols-3 features-grid">
            <div className="feature-card card">
              <div className="feature-icon">
                <FaLightbulb />
              </div>
              <h3 className="feature-title">Smart Recommendations</h3>
              <p className="feature-description">
                Our AI analyzes skill prerequisites, difficulty progressions, and course relationships
                to suggest the perfect next step in your learning journey.
              </p>
            </div>

            <div className="feature-card card">
              <div className="feature-icon">
                <FaChartLine />
              </div>
              <h3 className="feature-title">Learning Paths</h3>
              <p className="feature-description">
                Visualize your educational journey from beginner to expert.
                Follow curated paths or create your own based on your goals.
              </p>
            </div>

            <div className="feature-card card">
              <div className="feature-icon">
                <FaGraduationCap />
              </div>
              <h3 className="feature-title">Skill Mapping</h3>
              <p className="feature-description">
                Understand skill relationships and prerequisites.
                Build a comprehensive knowledge foundation systematically.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats">
        <div className="container">
          <div className="stats-grid grid grid-cols-3">
            <div className="stat-item">
              <div className="stat-number">1000+</div>
              <div className="stat-label">Courses Available</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">500+</div>
              <div className="stat-label">Skills Mapped</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">100+</div>
              <div className="stat-label">Learning Paths</div>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
};

export default Home;
