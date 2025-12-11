import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BookOpen, Lightbulb, Route, TrendingUp, Users, Zap } from "lucide-react";

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-16">
      {/* Hero Section */}
      <section className="text-center mb-20">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
          <Zap className="h-4 w-4" />
          <span>Knowledge-Based Learning Recommendations</span>
        </div>
        
        <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6 bg-gradient-to-r from-slate-900 to-slate-700 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
          Your Personalized Learning Journey
        </h1>
        
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
          Discover courses tailored to your skills and goals. Navigate from beginner to expert with intelligent, 
          graph-based course recommendations and personalized learning paths.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button size="lg" asChild className="text-lg px-8">
            <Link href="/learning-path">
              <Lightbulb className="mr-2 h-5 w-5" />
              Get Started
            </Link>
          </Button>
          
          <Button size="lg" variant="outline" asChild className="text-lg px-8">
            <Link href="/courses">
              <BookOpen className="mr-2 h-5 w-5" />
              Browse Courses
            </Link>
          </Button>
        </div>
      </section>

      {/* Features Section */}
      <section className="mb-20">
        <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          <Card className="border-2 hover:border-primary/50 transition-colors">
            <CardHeader>
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                <Users className="h-6 w-6 text-primary" />
              </div>
              <CardTitle>1. Tell Us About You</CardTitle>
              <CardDescription>
                Share your current skills, learning goals, and available time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Our system builds a knowledge base from your input to understand 
                your unique learning context and requirements.
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-primary/50 transition-colors">
            <CardHeader>
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                <TrendingUp className="h-6 w-6 text-primary" />
              </div>
              <CardTitle>2. AI Analysis</CardTitle>
              <CardDescription>
                Knowledge-based reasoning meets graph intelligence
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Our rule engine applies expert knowledge to analyze course relationships, 
                prerequisites, and skill progressions in Neo4j.
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-primary/50 transition-colors">
            <CardHeader>
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                <Route className="h-6 w-6 text-primary" />
              </div>
              <CardTitle>3. Your Learning Path</CardTitle>
              <CardDescription>
                Get a personalized, step-by-step roadmap
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Receive ranked course learning path recommendations with clear explanations 
                showing why each course fits your journey.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* CTA Section */}
      <section className="text-center bg-gradient-to-r from-primary/10 to-primary/5 rounded-2xl p-12">
        <h2 className="text-3xl font-bold mb-4">Ready to Start Learning?</h2>
        <p className="text-lg text-muted-foreground mb-8 max-w-xl mx-auto">
          Get personalized course recommendations powered by AI and knowledge graphs
        </p>
        <Button size="lg" asChild>
          <Link href="/learning-path">
            <Lightbulb className="mr-2 h-5 w-5" />
            Get Started Now
          </Link>
        </Button>
      </section>
    </div>
  );
}
