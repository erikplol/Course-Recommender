"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { getNextCourses } from "@/lib/api";
import { toast } from "sonner";
import { Sparkles, Target, Clock, TrendingUp, BookOpen, Star, Users, Award, ExternalLink, ArrowRight, Building2 } from "lucide-react";

export default function LearningPathPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [topic, setTopic] = useState("");
  const [currentSkills, setCurrentSkills] = useState("");
  const [learningGoals, setLearningGoals] = useState("");
  const [availableTime, setAvailableTime] = useState("3");
  const [userLevel, setUserLevel] = useState("3");
  const [recommendations, setRecommendations] = useState<any[]>([]);

  const handleGetPath = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!topic.trim()) {
      toast.error("Please enter what you want to learn");
      return;
    }

    setLoading(true);
    setRecommendations([]);
    
    try {
      const response = await getNextCourses(topic, {
        userLevel: parseInt(userLevel),
        currentSkills: currentSkills.split(",").map(s => s.trim()).filter(Boolean),
        learningGoals: learningGoals.split(",").map(s => s.trim()).filter(Boolean),
        availableTime: parseInt(availableTime),
      });
      
      setRecommendations(response.recommendations || []);
      toast.success("Your personalized learning path is ready!");
    } catch (error: any) {
      console.error("Error:", error);
      toast.error("Failed to generate learning path. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    const level = difficulty?.toLowerCase() || "";
    if (level.includes("beginner") || level.includes("introductory")) return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    if (level.includes("intermediate") || level.includes("mixed")) return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
    if (level.includes("advanced")) return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
    return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200";
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Hero Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
            <Sparkles className="h-8 w-8 text-primary" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-3">
            Build Your Learning Path
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Tell us what you want to learn and we'll create a personalized course path tailored to your goals
          </p>
        </div>

        {/* Input Form */}
        <Card className="mb-8 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Your Learning Goals
            </CardTitle>
            <CardDescription>
              Share your learning objectives and we'll recommend the perfect course sequence
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleGetPath} className="space-y-6">
              {/* Main Topic */}
              <div className="space-y-2">
                <Label htmlFor="topic" className="text-base font-semibold">
                  What do you want to learn? *
                </Label>
                <Input
                  id="topic"
                  placeholder="e.g., Machine Learning, Web Development, Data Science"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  required
                  className="text-base"
                />
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Current Skills */}
                <div className="space-y-2">
                  <Label htmlFor="skills" className="text-base font-semibold">
                    Your Current Skills
                  </Label>
                  <Input
                    id="skills"
                    placeholder="e.g., Python, Statistics"
                    value={currentSkills}
                    onChange={(e) => setCurrentSkills(e.target.value)}
                    className="text-base"
                  />
                  <p className="text-xs text-muted-foreground">Separate with commas</p>
                </div>

                {/* Learning Goals */}
                <div className="space-y-2">
                  <Label htmlFor="goals" className="text-base font-semibold">
                    What You Want to Achieve
                  </Label>
                  <Input
                    id="goals"
                    placeholder="e.g., Deep Learning, AI"
                    value={learningGoals}
                    onChange={(e) => setLearningGoals(e.target.value)}
                    className="text-base"
                  />
                  <p className="text-xs text-muted-foreground">Separate with commas</p>
                </div>
              </div>

              {/* Available Time */}
              <div className="space-y-2">
                <Label htmlFor="time" className="text-base font-semibold">
                  Time You Can Commit
                </Label>
                <Select value={availableTime} onValueChange={setAvailableTime}>
                  <SelectTrigger className="text-base">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">1 month</SelectItem>
                    <SelectItem value="2">2 months</SelectItem>
                    <SelectItem value="3">3 months</SelectItem>
                    <SelectItem value="6">6 months</SelectItem>
                    <SelectItem value="12">1 year</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button type="submit" className="w-full" size="lg" disabled={loading}>
                {loading ? (
                  <>
                    <Sparkles className="mr-2 h-5 w-5 animate-spin" />
                    Creating Your Path...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-5 w-5" />
                    Generate My Learning Path
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Results */}
        {recommendations.length > 0 && (
          <div className="space-y-8">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <TrendingUp className="h-6 w-6 text-primary" />
                  Your Personalized Learning Path
                </h2>
                <p className="text-muted-foreground mt-1">
                  {recommendations.length} courses recommended for your journey
                </p>
              </div>
            </div>

            {/* Graph View - Snake Pattern */}
            <Card className="p-6 bg-gradient-to-br from-primary/5 to-background">
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">Learning Journey Overview</h3>
                <p className="text-sm text-muted-foreground">Follow this path from start to finish</p>
              </div>
              
              <div className="relative space-y-8">
                {/* Group courses in rows of 3 for snake pattern */}
                {Array.from({ length: Math.ceil(recommendations.length / 3) }, (_, rowIndex) => {
                  const startIdx = rowIndex * 3;
                  const rowCourses = recommendations.slice(startIdx, startIdx + 3);
                  const isReversed = rowIndex % 2 === 1; // Alternate direction for snake effect
                  const displayCourses = isReversed ? [...rowCourses].reverse() : rowCourses;
                  
                  return (
                    <div key={rowIndex}>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                        {displayCourses.map((course, colIndex) => {
                          const actualIndex = isReversed 
                            ? startIdx + (2 - colIndex) 
                            : startIdx + colIndex;
                          
                          return (
                            <div key={actualIndex} className="relative">
                              {/* Course Node */}
                              <div 
                                className="group cursor-pointer h-full"
                                onClick={() => router.push(`/courses/${encodeURIComponent(course.title)}`)}
                              >
                                <div className="bg-background border-2 border-primary rounded-lg p-4 shadow-md hover:shadow-lg transition-all hover:scale-[1.02] h-full flex flex-col">
                                  <div className="flex items-start gap-3 mb-3">
                                    <div className="w-10 h-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-base flex-shrink-0">
                                      {actualIndex + 1}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm font-semibold mb-2 leading-tight">{course.title}</p>
                                      <div className="flex items-center gap-2 flex-wrap">
                                        {course.rating && (
                                          <div className="flex items-center gap-1 text-xs">
                                            <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                                            <span className="font-medium">{typeof course.rating === 'number' ? course.rating.toFixed(1) : course.rating}</span>
                                          </div>
                                        )}
                                        {course.students && (
                                          <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                            <Users className="h-3 w-3" />
                                            <span>{course.students >= 1000 ? `${(course.students / 1000).toFixed(1)}K` : course.students}</span>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                  <Badge 
                                    variant="secondary" 
                                    className={`${getDifficultyColor(course.difficulty || course.difficulty_level)} text-xs w-full justify-center mt-auto`}
                                  >
                                    {course.difficulty || course.difficulty_level || 'Mixed'}
                                  </Badge>
                                </div>
                              </div>
                              
                              {/* Horizontal Arrow*/}
                              {!isReversed && colIndex < rowCourses.length - 1 && actualIndex < recommendations.length - 1 && (
                                <div className="hidden md:block absolute top-1/2 -right-[42px] transform -translate-y-1/2 z-10">
                                  <div className="bg-primary/10 rounded-full p-1">
                                    <ArrowRight className="h-7 w-7 text-primary stroke-[2.5]" />
                                  </div>
                                </div>
                              )}
                              {isReversed && colIndex > 0 && actualIndex < recommendations.length - 1 && (
                                <div className="hidden md:block absolute top-1/2 -left-[42px] transform -translate-y-1/2 rotate-180 z-10">
                                  <div className="bg-primary/10 rounded-full p-1">
                                    <ArrowRight className="h-7 w-7 text-primary stroke-[2.5]" />
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                      
                      {/* Downward Arrow*/}
                      {rowIndex < Math.ceil(recommendations.length / 3) - 1 && startIdx + 2 < recommendations.length && (
                        <div className="hidden md:grid grid-cols-3 gap-10 mt-4">
                          {!isReversed ? (
                            <>
                              <div></div>
                              <div></div>
                              <div className="flex justify-center">
                                <div className="transform rotate-90">
                                  <div className="bg-primary/10 rounded-full p-1">
                                    <ArrowRight className="h-7 w-7 text-primary stroke-[2.5]" />
                                  </div>
                                </div>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="flex justify-center">
                                <div className="transform rotate-90">
                                  <div className="bg-primary/10 rounded-full p-1">
                                    <ArrowRight className="h-7 w-7 text-primary stroke-[2.5]" />
                                  </div>
                                </div>
                              </div>
                              <div></div>
                              <div></div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </Card>

            {/* Detailed List View */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Detailed Course Information</h3>
              <div className="grid gap-4">
              {recommendations.map((course, index) => (
                <Card key={index} className="overflow-hidden hover:shadow-lg transition-all cursor-pointer group">
                  <CardContent className="p-6">
                    <div className="flex items-start gap-4">
                      {/* Step Number */}
                      <div className="flex-shrink-0">
                        <div className="w-12 h-12 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-lg shadow-md">
                          {index + 1}
                        </div>
                      </div>

                      {/* Course Details */}
                      <div className="flex-1 space-y-3">
                        <div>
                          <h3 className="text-xl font-bold mb-2">
                            {course.title}
                          </h3>
                          <p className="text-sm text-muted-foreground mb-3 flex items-center gap-1">
                            <Building2 className="h-3.5 w-3.5" />
                            {course.organization}
                          </p>
                        </div>

                        {/* Course Stats - Prominent Display */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 bg-muted/30 rounded-lg border">
                          {(course.difficulty || course.difficulty_level) && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground mb-1">Level</p>
                              <Badge variant="secondary" className={`${getDifficultyColor(course.difficulty || course.difficulty_level)} w-full justify-center text-xs`}>
                                {course.difficulty || course.difficulty_level || 'Mixed'}
                              </Badge>
                            </div>
                          )}
                          {course.rating && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground mb-1">Rating</p>
                              <div className="flex items-center justify-center gap-1 font-semibold">
                                <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                                <span className="text-sm">{typeof course.rating === 'number' ? course.rating.toFixed(1) : course.rating}</span>
                              </div>
                            </div>
                          )}
                          {(course.students || course.enrolled_student_count) && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground mb-1">Students</p>
                              <div className="flex items-center justify-center gap-1 font-semibold text-sm">
                                <Users className="h-4 w-4 text-primary" />
                                <span>
                                  {((course.students || course.enrolled_student_count) >= 1000) 
                                    ? `${((course.students || course.enrolled_student_count) / 1000).toFixed(1)}K`
                                    : (course.students || course.enrolled_student_count).toLocaleString()}
                                </span>
                              </div>
                            </div>
                          )}
                          {(course.duration || course.estimated_duration) && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground mb-1">Duration</p>
                              <div className="flex items-center justify-center gap-1 font-semibold text-sm">
                                <Clock className="h-4 w-4 text-primary" />
                                <span className="truncate">{course.duration || course.estimated_duration}</span>
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Certificate Badge */}
                        {course.certificate_type && (
                          <Badge variant="outline" className="flex items-center gap-1 w-fit">
                            <Award className="h-3 w-3" />
                            {course.certificate_type}
                          </Badge>
                        )}

                        {/* Why This Course */}
                        {course.explanation && (
                          <div className="bg-muted/50 rounded-lg p-3 border-l-4 border-primary">
                            <p className="text-sm font-medium mb-2 flex items-center gap-1">
                              <TrendingUp className="h-4 w-4" />
                              Why this course?
                            </p>
                            <ul className="text-sm text-muted-foreground space-y-1">
                              {course.explanation.split('•').filter((point: string) => point.trim()).map((point: string, i: number) => (
                                <li key={i} className="flex items-start gap-2">
                                  <span className="text-primary mt-0.5">•</span>
                                  <span>{point.trim()}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Skills */}
                        {course.skills && course.skills.length > 0 && (
                          <div>
                            <p className="text-sm font-medium mb-2 flex items-center gap-1">
                              <BookOpen className="h-4 w-4" />
                              What you'll learn:
                            </p>
                            <div className="flex flex-wrap gap-1">
                              {course.skills.slice(0, 6).map((skill: string, i: number) => (
                                <Badge key={i} variant="secondary" className="text-xs">
                                  {skill}
                                </Badge>
                              ))}
                              {course.skills.length > 6 && (
                                <Badge variant="secondary" className="text-xs">
                                  +{course.skills.length - 6} more
                                </Badge>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex gap-2 pt-2">
                          <Button
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              router.push(`/courses/${encodeURIComponent(course.title)}`);
                            }}
                          >
                            <ArrowRight className="h-4 w-4 mr-1" />
                            View Details
                          </Button>
                          {course.url && (
                            <Button
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                window.open(course.url, '_blank');
                              }}
                              variant="outline"
                            >
                              <ExternalLink className="h-4 w-4 mr-1" />
                              Go to Course
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!loading && recommendations.length === 0 && (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                <Target className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Ready to Start Learning?</h3>
              <p className="text-muted-foreground max-w-md">
                Fill in the form above to get personalized course recommendations based on your goals and experience
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
