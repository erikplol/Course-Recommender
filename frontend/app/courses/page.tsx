"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getCourses } from "@/lib/api";
import { Course } from "@/lib/types";
import { toast } from "sonner";
import { Search, Star, Users, Clock, ExternalLink, ArrowRight, SlidersHorizontal, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";

export default function CoursesPage() {
  const router = useRouter();
  const [courses, setCourses] = useState<Course[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const ITEMS_PER_PAGE = 20;
  const [showFilters, setShowFilters] = useState(false);
  
  // Filter states
  const [difficulty, setDifficulty] = useState("all");
  const [duration, setDuration] = useState("all");
  const [sortBy, setSortBy] = useState("rating");
  const [sortOrder, setSortOrder] = useState("desc");
  const [minRating, setMinRating] = useState("any");

  useEffect(() => {
    loadCourses();
  }, [page, difficulty, duration, sortBy, sortOrder, minRating]);

  const loadCourses = async () => {
    setLoading(true);
    try {
      const params: any = { page, limit: ITEMS_PER_PAGE };
      
      if (search) params.search = search;
      if (difficulty !== "all") params.difficulty = difficulty;
      if (duration !== "all") params.duration = duration;
      if (minRating !== "any") params.minRating = minRating;
      params.sortBy = sortBy;
      params.sortOrder = sortOrder;
      
      const response = await getCourses(params);
      const fetchedCourses = response.courses || [];
      setCourses(fetchedCourses);
      setHasMore(fetchedCourses.length === ITEMS_PER_PAGE);
    } catch (error) {
      console.error("Error:", error);
      toast.error("Failed to load courses");
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    loadCourses();
  };

  const resetFilters = () => {
    setSearch("");
    setDifficulty("all");
    setDuration("all");
    setSortBy("rating");
    setSortOrder("desc");
    setMinRating("any");
    setPage(1);
  };

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">Browse Courses</h1>
        <p className="text-xl text-muted-foreground">
          Explore thousands of courses from top organizations
        </p>
      </div>

      {/* Search Bar with Filter Button */}
      <div className="mb-6">
        <form onSubmit={handleSearch} className="flex gap-2 mb-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search courses by title, summary, or skills..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-10"
            />
          </div>
          <Button 
            type="button" 
            variant="outline"
            onClick={() => setShowFilters(!showFilters)}
            className="gap-2"
          >
            <SlidersHorizontal className="h-4 w-4" />
            Filters
            {(difficulty !== "all" || duration !== "all" || minRating !== "any") && (
              <Badge variant="default" className="ml-1 px-1.5 py-0 h-5 min-w-5">
                {[difficulty !== "all", duration !== "all", minRating !== "any"].filter(Boolean).length}
              </Badge>
            )}
          </Button>
          <Button type="submit">Search</Button>
        </form>

        {/* Collapsible Filters */}
        {showFilters && (
          <Card className="border-primary/20">
            <CardHeader className="pb-1">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <SlidersHorizontal className="h-5 w-5" />
                  Filters
                </CardTitle>
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setShowFilters(false)}
                >
                  Close
                </Button>
              </div>
            </CardHeader>
            <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* Sort By */}
            <div className="space-y-0">
              <label className="text-sm font-medium">Sort By</label>
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="rating">Rating</SelectItem>
                  <SelectItem value="students">Students</SelectItem>
                  <SelectItem value="title">Title</SelectItem>
                  <SelectItem value="duration">Duration</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Sort Order */}
            <div className="space-y-0">
              <label className="text-sm font-medium">Order</label>
              <Select value={sortOrder} onValueChange={setSortOrder}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="desc">Descending</SelectItem>
                  <SelectItem value="asc">Ascending</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Difficulty Level */}
            <div className="space-y-0">
              <label className="text-sm font-medium">Difficulty</label>
              <Select value={difficulty} onValueChange={setDifficulty}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="Beginner">Beginner</SelectItem>
                  <SelectItem value="Intermediate">Intermediate</SelectItem>
                  <SelectItem value="Advanced">Advanced</SelectItem>
                  <SelectItem value="Mixed">Mixed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Duration */}
            <div className="space-y-0">
              <label className="text-sm font-medium">Duration</label>
              <Select value={duration} onValueChange={setDuration}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Durations</SelectItem>
                  <SelectItem value="short">Short (≤1 month)</SelectItem>
                  <SelectItem value="medium">Medium (2-4 months)</SelectItem>
                  <SelectItem value="long">Long (5+ months)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Minimum Rating */}
            <div className="space-y-0">
              <label className="text-sm font-medium">Min Rating</label>
              <Select value={minRating} onValueChange={setMinRating}>
                <SelectTrigger>
                  <SelectValue placeholder="Any Rating" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="any">Any Rating</SelectItem>
                  <SelectItem value="4.5">4.5+ Stars</SelectItem>
                  <SelectItem value="4.0">4.0+ Stars</SelectItem>
                  <SelectItem value="3.5">3.5+ Stars</SelectItem>
                  <SelectItem value="3.0">3.0+ Stars</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

                  <div className="flex justify-end mt-4">
                <Button variant="outline" onClick={resetFilters} size="sm">
                  Reset Filters
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {loading ? (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-3/4" />
                <Skeleton className="h-4 w-full" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-20 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {courses.map((course, index) => (
            <Card 
              key={index} 
              className="hover:shadow-lg transition-all cursor-pointer group"
              onClick={() => router.push(`/courses/${encodeURIComponent(course.title)}`)}
            >
              <CardHeader>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <Badge variant="secondary">{course.difficulty}</Badge>
                  <div className="flex items-center gap-1 text-sm">
                    <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                    <span>{course.rating.toFixed(1)}</span>
                  </div>
                </div>
                <CardTitle className="line-clamp-2 group-hover:text-primary transition-colors">
                  {course.title}
                </CardTitle>
                <CardDescription className="line-clamp-2">
                  {course.summary}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap gap-3 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Users className="h-4 w-4" />
                    <span>{course.students.toLocaleString()}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    <span>{course.duration}</span>
                  </div>
                </div>
                
                <div className="text-sm text-muted-foreground">
                  {course.organization}
                </div>

                {course.skills && course.skills.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {course.skills.slice(0, 3).map((skill, i) => (
                      <Badge key={i} variant="outline" className="text-xs">
                        {skill}
                      </Badge>
                    ))}
                    {course.skills.length > 3 && (
                      <Badge variant="outline" className="text-xs">
                        +{course.skills.length - 3}
                      </Badge>
                    )}
                  </div>
                )}

                <div className="flex gap-2">
                  <Button 
                    className="flex-1" 
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      router.push(`/courses/${encodeURIComponent(course.title)}`);
                    }}
                  >
                    <ArrowRight className="mr-2 h-4 w-4" />
                    View Details
                  </Button>
                  <Button 
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      window.open(course.url, '_blank');
                    }}
                  >
                    <ExternalLink className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {!loading && courses.length === 0 && (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No courses found. Try a different search.</p>
        </div>
      )}

      {!loading && courses.length > 0 && (
        <div className="mt-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            {/* Results Info */}
            <div className="text-sm text-muted-foreground">
              Showing {((page - 1) * ITEMS_PER_PAGE) + 1} - {((page - 1) * ITEMS_PER_PAGE) + courses.length} courses
              {page === 1 && courses.length < ITEMS_PER_PAGE && (
                <span> (all results)</span>
              )}
            </div>

            {/* Pagination Controls */}
            <div className="flex items-center gap-2">
              {/* First Page */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage(1)}
                disabled={page === 1}
                className="hidden sm:flex"
              >
                <ChevronsLeft className="h-4 w-4" />
              </Button>

              {/* Previous Page */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage(Math.max(1, page - 1))}
                disabled={page === 1}
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Previous
              </Button>

              {/* Page Numbers */}
              <div className="hidden sm:flex items-center gap-1">
                {page > 2 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page - 2)}
                  >
                    {page - 2}
                  </Button>
                )}
                {page > 1 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page - 1)}
                  >
                    {page - 1}
                  </Button>
                )}
                <Button
                  variant="default"
                  size="sm"
                  disabled
                >
                  {page}
                </Button>
                {hasMore && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page + 1)}
                  >
                    {page + 1}
                  </Button>
                )}
                {hasMore && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page + 2)}
                  >
                    {page + 2}
                  </Button>
                )}
              </div>

              {/* Mobile Page Indicator */}
              <div className="sm:hidden px-3 py-1.5 text-sm font-medium border rounded-md bg-background">
                Page {page}
              </div>

              {/* Next Page */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage(page + 1)}
                disabled={!hasMore}
              >
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>

              {/* Last Page (approximate) */}
              {hasMore && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(page + 5)}
                  className="hidden sm:flex"
                >
                  <ChevronsRight className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>

          {/* Scroll to top helper */}
          <div className="text-center mt-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-muted-foreground"
            >
              Scroll to top
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
