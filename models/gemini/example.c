// examples.c
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>  // For SIMD intrinsics
#include <stdarg.h>     // For variable arguments

// 1) Basic‐type function (takes and returns an int)
int square(int x) {
    return x * x;
}

// 2) Multiple basic‐type parameters
int sum(int a, int b, int c) {
    return a + b + c;
}

// 3) Pointer parameters (swap two ints)
void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// 4) Passing/returning a struct by value
struct Point {
    int x, y;
};
void printPoint(struct Point p) {
    printf("Point(%d, %d)\n", p.x, p.y);
}

// 5) Array parameter (decays to pointer)
void printArray(int arr[], int size) {
    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// 6) Array of structs
struct Person {
    char name[50];
    int age;
};
void printPeople(struct Person people[], int count) {
    printf("People:\n");
    for (int i = 0; i < count; i++) {
        printf("  %s, age %d\n",
               people[i].name,
               people[i].age);
    }
}

// 7) Floating-point parameters
double calculateDistance(double x1, double y1, double x2, double y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// 8) Mixed integer and floating-point parameters
double weightedAverage(int a, double b, int c, double d) {
    return (a + b + c + d) / 4.0;
}

// 9) SIMD vector operations (SSE)
__m128 addVectors(__m128 a, __m128 b) {
    return _mm_add_ps(a, b);
}

// 10) SIMD vector operations (AVX)
__m256 addVectors256(__m256 a, __m256 b) {
    return _mm256_add_ps(a, b);
}

// 11) Complex number operations
struct Complex {
    double real;
    double imag;
};

int global_scale = 10;

struct Complex multiplyComplex(struct Complex a, struct Complex b) {
    struct Complex result;
    result.real = (a.real * b.real - a.imag * b.imag) * global_scale;
    result.imag = (a.real * b.imag + a.imag * b.real) * global_scale;
    return result;
}

// 12) Variable arguments
double average(int count, ...) {
    va_list args;
    va_start(args, count);
    double sum = 0;
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, double);
    }
    va_end(args);
    return sum / count;
}

int main(void) {
    // --- 1) Basic‐type function ---
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    int v = a;
    
    printf("square(%d) = %d\n", v, square(v));

    // --- 2) Multiple basic‐type parameters ---
    printf("sum(1, 2, 3) = %d\n", sum(a, b, c));

    // --- 3) Pointer parameters ---
    printf("Before swap: a=%d, b=%d\n", a, b);
    swap(&a, &b);
    printf("After  swap: a=%d, b=%d\n", a, b);

    // --- 4) Struct by value ---
    struct Point p = { .x = a, .y = b };
    printPoint(p);

    // --- 5) Array parameter ---
    int arr[] = { a, b, c, v, 1 };
    printArray(arr, sizeof(arr)/sizeof(arr[0]));

    // --- 6) Array of structs ---
    struct Person people[2];
    strcpy(people[0].name, "Alice");
    people[0].age = a;
    strcpy(people[1].name, "Bob");
    people[1].age = b;
    printPeople(people, 2);

    // --- 7) Floating-point parameters ---
    double dist = calculateDistance(1.0, 2.0, 4.0, 6.0);
    printf("Distance: %f\n", dist);

    // --- 8) Mixed parameters ---
    double avg = weightedAverage(1, 2.5, 3, 4.5);
    printf("Weighted average: %f\n", avg);

    // --- 9) SIMD operations (SSE) ---
    __m128 vec1 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
    __m128 vec2 = _mm_set_ps(5.0f, 6.0f, 7.0f, 8.0f);
    __m128 result = addVectors(vec1, vec2);
    float result_array[4];
    _mm_store_ps(result_array, result);
    printf("Vector addition result: %f %f %f %f\n", 
           result_array[0], result_array[1], result_array[2], result_array[3]);

    // --- 10) SIMD operations (AVX) ---
    __m256 vec1_256 = _mm256_set_ps(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    __m256 vec2_256 = _mm256_set_ps(9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
    __m256 result_256 = addVectors256(vec1_256, vec2_256);
    float result_array_256[8];
    _mm256_store_ps(result_array_256, result_256);
    printf("AVX Vector addition result: %f %f %f %f %f %f %f %f\n",
           result_array_256[0], result_array_256[1], result_array_256[2], result_array_256[3],
           result_array_256[4], result_array_256[5], result_array_256[6], result_array_256[7]);

    // --- 11) Complex number operations ---
    struct Complex c1 = {1.0, 2.0};
    struct Complex c2 = {3.0, 4.0};
    struct Complex c3 = multiplyComplex(c1, c2);
    printf("Complex multiplication: %f + %fi\n", c3.real, c3.imag);

    // --- 12) Variable arguments ---
    double var_avg = average(4, 1.0, 2.0, 3.0, 4.0);
    printf("Variable argument average: %f\n", var_avg);

    return 0;
}