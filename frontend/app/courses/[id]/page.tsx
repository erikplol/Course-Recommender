"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getCourses } from "@/lib/api";
import { toast } from "sonner";
import { 
  ArrowLeft, 
  ExternalLink, 
  Star, 
  Users, 
  Clock, 
  BookOpen, 
  Award, 
  Building2,
  Target,
  TrendingUp
} from "lucide-react";

export default function CourseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [course, setCourse] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCourse();
  }, [params.id]);

  const loadCourse = async () => {
    setLoading(true);
    try {
      // Decode the course title from the URL
      const courseTitle = decodeURIComponent(params.id as string);
      
      // Search for the specific course
      const response = await getCourses({ search: courseTitle, limit: 50 });
      
      // Find the exact match or closest match
      const foundCourse = response.courses?.find(
        (c: any) => c.title.toLowerCase() === courseTitle.toLowerCase()
      ) || response.courses?.[0];

      if (foundCourse) {
        setCourse(foundCourse);
      } else {
        toast.error("Course not found");
        router.push("/courses");
      }
    } catch (error) {
      console.error("Error:", error);
      toast.error("Failed to load course details");
    } finally {
      setLoading(false);
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    const level = difficulty?.toLowerCase() || "";
    if (level.includes("beginner") || level.includes("introductory")) 
      return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    if (level.includes("intermediate") || level.includes("mixed")) 
      return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
    if (level.includes("advanced")) 
      return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
    return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200";
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12 max-w-5xl">
        <Skeleton className="h-8 w-32 mb-8" />
        <Card>
          <CardHeader>
            <Skeleton className="h-10 w-3/4 mb-4" />
            <Skeleton className="h-6 w-1/2" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-32 w-full mb-4" />
            <Skeleton className="h-20 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!course) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Back Button */}
        <Button
          variant="ghost"
          onClick={() => router.back()}
          className="mb-6"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>

        {/* Main Course Card */}
        <Card className="mb-6 shadow-lg">
          <CardHeader>
            <div className="flex flex-wrap gap-2 mb-4">
              <Badge variant="secondary" className={getDifficultyColor(course.difficulty)}>
                {course.difficulty}
              </Badge>
              {course.certificate_type && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <Award className="h-3 w-3" />
                  {course.certificate_type}
                </Badge>
              )}
              {course.rating && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                  {course.rating.toFixed(1)}
                </Badge>
              )}
            </div>
            
            <CardTitle className="text-3xl md:text-4xl mb-3">
              {course.title}
            </CardTitle>
            
            <CardDescription className="flex items-center gap-2 text-base">
              <Building2 className="h-4 w-4" />
              {course.organization}
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Course Stats */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
              {course.enrolled_student_count && (
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Students Enrolled</p>
                  <p className="font-semibold flex items-center gap-1 text-lg">
                    <Users className="h-5 w-5 text-primary" />
                    {course.enrolled_student_count?.toLocaleString() || course.students?.toLocaleString()}
                  </p>
                </div>
              )}
              {(course.estimated_duration || course.duration) && (
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Duration</p>
                  <p className="font-semibold flex items-center gap-1 text-lg">
                    <Clock className="h-5 w-5 text-primary" />
                    {course.estimated_duration || course.duration}
                  </p>
                </div>
              )}
              {course.rating && (
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Rating</p>
                  <p className="font-semibold flex items-center gap-1 text-lg">
                    <Star className="h-5 w-5 text-yellow-400 fill-yellow-400" />
                    {course.rating.toFixed(1)} / 5.0
                  </p>
                </div>
              )}
            </div>

            {/* Description */}
            {(course.description || course.summary) && (
              <div>
                <h3 className="text-xl font-semibold mb-3 flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  About This Course
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  {course.description || course.summary}
                </p>
              </div>
            )}

            {/* Skills */}
            {course.skills && course.skills.length > 0 && (
              <div>
                <h3 className="text-xl font-semibold mb-3 flex items-center gap-2">
                  <BookOpen className="h-5 w-5" />
                  Skills You'll Gain
                </h3>
                <div className="flex flex-wrap gap-2">
                  {course.skills.map((skill: string, i: number) => (
                    <Badge key={i} variant="secondary" className="text-sm py-1 px-3">
                      {skill}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Prerequisites */}
            {course.prerequisites && course.prerequisites.length > 0 && (
              <div>
                <h3 className="text-xl font-semibold mb-3 flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Prerequisites
                </h3>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  {course.prerequisites.map((prereq: string, i: number) => (
                    <li key={i}>{prereq}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Certificate */}
            {course.certificate_type && (
              <div className="bg-primary/5 rounded-lg p-4 border-l-4 border-primary">
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <Award className="h-5 w-5" />
                  Certificate Available
                </h4>
                <p className="text-sm text-muted-foreground">
                  Earn a {course.certificate_type} certificate upon completion
                </p>
              </div>
            )}

            {/* Course Link */}
            {course.url && (
              <Button
                className="w-full"
                size="lg"
                onClick={() => window.open(course.url, '_blank')}
              >
                <ExternalLink className="h-5 w-5 mr-2" />
                Go to Course Page
              </Button>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
