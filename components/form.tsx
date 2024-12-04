"use client";

import { toast } from "sonner";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";

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

const formSchema = z.object({
  name_7577462224: z.string().optional(),
  name_4440762964: z.string().optional(),
  name_7948944962: z.number().min(0).max(1000).optional(),
  name_6717152066: z.array(z.string()).nonempty("Please select at least one item").optional(),
  name_7441719795: z.string().optional(),
});

export default function MyForm() {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema), 
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    try {
      console.log(values);
      toast(
        <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4">
          <code className="text-white">{JSON.stringify(values, null, 2)}</code>
        </pre>
      );
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
          <h1 className="text-3xl font-bold text-center text-gray-800">Book Recommendation Form</h1>
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

          {/* Number of Pages */}
          <FormField
            control={form.control}
            name="name_7948944962"
            defaultValue={0}
            render={({ field: { value, onChange } }) => (
              <FormItem>
                <FormLabel className="text-lg font-medium">Number of Pages - {value}</FormLabel>
                <FormControl>
                  <Slider
                    min={0}
                    max={1000}
                    step={20}
                    defaultValue={[5]}
                    onValueChange={(vals) => onChange(vals[0])}
                  />
                </FormControl>
                <FormDescription className="text-sm text-gray-500">
                  Adjust the range by sliding.
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
                        <MultiSelectorItem value={"Romance"}>Romance</MultiSelectorItem>
                        <MultiSelectorItem value={"Comedy"}>Comedy</MultiSelectorItem>
                        <MultiSelectorItem value={"Fantasy"}>Fantasy</MultiSelectorItem>
                        <MultiSelectorItem value={"Mystery"}>Mystery</MultiSelectorItem>
                        <MultiSelectorItem value={"Scifi"}>Sci-fi</MultiSelectorItem>
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
    </div>
  );
}
