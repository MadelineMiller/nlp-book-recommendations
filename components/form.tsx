"use client";

import { toast } from "sonner";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useState } from "react";
import StarRatings from "react-star-ratings";

import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "@/components/ui/multi-select";
import { Textarea } from "@/components/ui/textarea";
import { genres } from "./genre-list";

import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";

// Define the type for a single recommendation
type Recommendation = {
  Title: string;
  Author: string;
  Decades: string;
  Genre: string;
  Description: string;
  Rating?: number; // Optional rating field
};

const formSchema = z.object({
  name_7577462224: z.string().optional(),
  name_4440762964: z.string().optional(),
  name_7948944962: z.number().optional(),
  name_6717152066: z.array(z.string()).nonempty("Please select at least one item"),
  name_7441719795: z.string().optional(),
});

export default function MyForm() {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
  });

  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isModalOpen, setModalOpen] = useState(false);

  const handleRatingChange = (index: number, rating: number) => {
    setRecommendations((prevRecommendations) => {
      const updatedRecommendations = [...prevRecommendations];
      updatedRecommendations[index].Rating = rating;
      return updatedRecommendations;
    });
    console.log(`Rating for ${recommendations[index]?.Title}: ${rating}`);
  };

  function onSubmit(values: z.infer<typeof formSchema>) {
    try {
      const queryParams = new URLSearchParams({
        title: encodeURIComponent(values.name_7577462224 || ""),
        author: encodeURIComponent(values.name_4440762964 || ""),
        decades: encodeURIComponent(values.name_7948944962?.toString() || ""),
        genres: encodeURIComponent(values.name_6717152066.join(",")) || "",
        description: encodeURIComponent(values.name_7441719795 || ""),
        top_n: "5",
      }).toString();

      console.log("Generated Query String:", queryParams);

      fetch(`http://127.0.0.1:5000/recommend?${queryParams}`)
        .then((response) => {
          if (!response.ok) {
            throw new Error("Failed to fetch recommendations.");
          }
          return response.json();
        })
        .then((data) => {
          console.log("Fetched Recommendations:", data);
          toast.success("Recommendations fetched successfully!");
          setRecommendations(data || []);
          setModalOpen(true);
        })
        .catch((error) => {
          console.error("Error fetching recommendations", error);
          toast.error("Failed to fetch recommendations. Please try again.");
        });
    } catch (error) {
      console.error("Form submission error", error);
      toast.error("Failed to submit the form. Please try again.");
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white flex items-center justify-center">
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(onSubmit)}
          className="space-y-8 w-full max-w-4xl mx-auto p-8 bg-white rounded-lg shadow-lg border border-gray-200"
        >
          <h1 className="text-3xl font-bold text-center text-gray-800">
            Book Recommendation Form
          </h1>
          <p className="text-center text-gray-600">
            Fill out the details below to get personalized book recommendations.
          </p>

          {/* Book Title */}
          <FormField
            control={form.control}
            name="name_7577462224"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="text-lg font-medium">Book Title</FormLabel>
                <FormControl>
                  <Input
                    placeholder="Enter the book title (e.g., 'Pride and Prejudice')"
                    {...field}
                    className="text-base p-4 border-gray-300 rounded-md"
                  />
                </FormControl>
                <FormDescription className="text-sm text-gray-500">
                  Recommend me something similar to this book.
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Author */}
          <FormField
            control={form.control}
            name="name_4440762964"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="text-lg font-medium">Author</FormLabel>
                <FormControl>
                  <Input
                    placeholder="Enter the author's name (e.g., 'Roald Dahl')"
                    {...field}
                    className="text-base p-4 border-gray-300 rounded-md"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Decade */}
          <FormField
            control={form.control}
            name="name_7948944962"
            defaultValue={1700}
            render={({ field: { value, onChange } }) => (
              <FormItem>
                <FormLabel className="text-lg font-medium">
                  From decade - {value}
                </FormLabel>
                <FormControl>
                  <Slider
                    min={1700}
                    max={2020}
                    step={10}
                    defaultValue={[5]}
                    onValueChange={(vals) => onChange(vals[0])}
                  />
                </FormControl>
                <FormDescription className="text-sm text-gray-500">
                  Adjust the decade by sliding.
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Genres */}
          <FormField
            control={form.control}
            name="name_6717152066"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="text-lg font-medium">Genres</FormLabel>
                <FormControl>
                  <MultiSelector
                    values={field.value ?? []}
                    onValuesChange={(newValues) => field.onChange(newValues)}
                    className="max-w-sm"
                  >
                    <MultiSelectorTrigger>
                      <MultiSelectorInput placeholder="Select genres" />
                    </MultiSelectorTrigger>
                    <MultiSelectorContent>
                      <MultiSelectorList>
                        {genres.map((genre) => (
                          <MultiSelectorItem key={genre} value={genre.toUpperCase()}>
                            {genre.charAt(0).toUpperCase() + genre.slice(1)}
                          </MultiSelectorItem>
                        ))}
                      </MultiSelectorList>
                    </MultiSelectorContent>
                  </MultiSelector>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Description */}
          <FormField
            control={form.control}
            name="name_7441719795"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="text-lg font-medium">Description</FormLabel>
                <FormControl>
                  <Textarea
                    placeholder="Enter keywords or related descriptions"
                    {...field}
                    className="resize-none text-base p-4 border-gray-300 rounded-md"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Submit Button */}
          <Button
            type="submit"
            className="w-full py-4 text-lg font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-md transition duration-200"
          >
            Submit
          </Button>
        </form>
      </Form>

      {/* Modal for Recommendations */}
      {isModalOpen && (
        <Dialog open={isModalOpen} onOpenChange={setModalOpen}>
          <DialogContent
            aria-describedby="recommendations-description"
            className="max-h-[80vh] overflow-y-auto p-6 rounded-lg shadow-lg"
          >
            <DialogTitle className="text-2xl font-bold text-center mb-4">
              Recommended Books
            </DialogTitle>
            <p id="recommendations-description" className="text-gray-500 text-center mb-6">
              Below are books tailored to your preferences.
            </p>
            {recommendations.length > 0 ? (
              <ul className="space-y-6">
                {recommendations.map((rec, index) => (
                  <li
                    key={index}
                    className="border rounded-lg shadow-sm p-4 hover:shadow-md transition-shadow"
                  >
                    <p className="text-lg font-semibold text-blue-800 capitalize">
                      {rec.Title}
                    </p>
                    <p className="text-gray-700 capitalize">
                      <strong>Author:</strong> {rec.Author || "Unknown"}
                    </p>
                    <p className="text-gray-700 capitalize">
                      <strong>Decades:</strong> {rec.Decades || "N/A"}
                    </p>
                    <p className="text-gray-700 capitalize">
                      <strong>Genre:</strong> {rec.Genre || "N/A"}
                    </p>
                    <p className="text-gray-700 mt-2">
                      <strong>Description:</strong>{" "}
                      {rec.Description
                        ? rec.Description.charAt(0).toUpperCase() +
                          rec.Description.slice(1).replace(/\s+/g, " ")
                        : "No description available."}
                    </p>
                    {/* Star Rating */}
                    <div className="mt-4">
                      <strong>Rate this Recommendation:</strong>
                      <StarRatings
                        rating={rec.Rating || 0}
                        starRatedColor="gold"
                        starHoverColor="gold"
                        numberOfStars={5}
                        name={`rating-${index}`}
                        changeRating={(rating) => handleRatingChange(index, rating)}
                        starDimension="24px"
                        starSpacing="4px"
                      />
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500 text-center">No recommendations available.</p>
            )}
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
