### QUESTION 1
#### Given a integer matrix (or 2D array) a[][] of dimensions n * m. Also, given another 2-D array query[][] of dimensions q * 4.

For each index 0 < i < query.length, find the sum of all the elements of the rectangular matrix whose top left corner is (query[i][0], query[i][1]) and bottom right corner is (query[i][2], query[i][3]).

Example - 

Input:
n = 3, m = 3, q = 2 
a[][] = [
          [ 1, 2, 3],
          [ 4, 5, 6],
          [ 7, 8, 9]
        ]

query[][] = [
               [0, 0, 2, 2]
               [1, 0, 2, 1]
            ]

Output:
45
24

Explanation:

The sum of all the elements in the matrix whose top left corner is (0, 0) and the bottom right corner is (2, 2) is 45.

The sum of all the elements in the matrix whose top left corner is (1, 0) and the bottom right corner is (2, 1) is 24.

```java

import java.util.*;

class Solution {
    ArrayList<Long> submatrixSum(long[][] a, int n, 
                                 int m, int[][] query, int q) {

        // Step 1: Create prefix sum matrix
        long[][] prefix = new long[n][m];

        // Fill prefix matrix
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){

                prefix[i][j] = a[i][j];

                if(i > 0)
                    prefix[i][j] += prefix[i-1][j];

                if(j > 0)
                    prefix[i][j] += prefix[i][j-1];

                if(i > 0 && j > 0)
                    prefix[i][j] -= prefix[i-1][j-1];
            }
        }

        // Step 2: Process queries
        ArrayList<Long> result = new ArrayList<>();

        for(int i = 0; i < q; i++){

            int r1 = query[i][0];
            int c1 = query[i][1];
            int r2 = query[i][2];
            int c2 = query[i][3];

            long sum = prefix[r2][c2];

            if(r1 > 0)
                sum -= prefix[r1-1][c2];

            if(c1 > 0)
                sum -= prefix[r2][c1-1];

            if(r1 > 0 && c1 > 0)
                sum += prefix[r1-1][c1-1];

            result.add(sum);
        }

        return result;
    }
}
```
### Question 2
Given a matrix a of size n*m which represents a park, there is some construction work needs to be done. You are also given q queries each query contains two numbers R and C, For every query we need to construct a footpath in the Rth row and Cth column, there is a cost of this construction, after the construction this path will divide the park into sections, and the cost of the construction is the sum of minimum value present in all the sections. You are asked to find this cost for all the queries.

Note: Elements present in queries array are according to 1-based indexing.

Example 1:

Input:
n=3
m=3
a={{1,2,3},{4,5,6},{7,8,9}}
q=1
queries={{2,2}}

Output:
20

Explanation:
For query {2,2}, after Footpath construction, park looks like:

1 * 3
* * *
7 * 9

Here star represents footpath, Sum of minimum value present in all the remaining sections is 1+3+7+9=20

Example 2:

Input:
n=3
m=4
a={{1,2,3,4},{5,6,7,8},{1,2,3,4}}
q=1
queries={{3,4}}

Output:
1

Explantion:
For Query {3,4}, after Footpath construction, park looks like:

1 2 3 *
5 6 7 *
* * * *

There is only one section, sum of minimum value present in all the remaining sections is 1.

Your Task:
You don't need to read input or print anything. Your task is to complete the function createFootpath() that takes n, m, a, q and queries as input and return an array of size q containing answer to every query as described in the problem statement.

Expected Time Complexity: O(n*m)
Expected Space Complexity: O(n*m)

Constraints:
1 <= n <= 105
1 <= m <= 105
1 <= n*m <= 105
1 <= a[i][j] <= 105
1 <= q <= 105
1 <= queries[i][0] <= n
1 <= queries[i][1] <= m

```java
class Solution {

    public static int[] createFootpath(int n, int m, int[][] a, int q, int[][] queries) {

        int[][] topLeft = new int[n][m];
        int[][] topRight = new int[n][m];
        int[][] bottomLeft = new int[n][m];
        int[][] bottomRight = new int[n][m];

        // 1️⃣ Top Left
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {

                topLeft[i][j] = a[i][j];

                if (i > 0)
                    topLeft[i][j] = Math.min(topLeft[i][j], topLeft[i - 1][j]);

                if (j > 0)
                    topLeft[i][j] = Math.min(topLeft[i][j], topLeft[i][j - 1]);
            }
        }

        // 2️⃣ Top Right
        for (int i = 0; i < n; i++) {
            for (int j = m - 1; j >= 0; j--) {

                topRight[i][j] = a[i][j];

                if (i > 0)
                    topRight[i][j] = Math.min(topRight[i][j], topRight[i - 1][j]);

                if (j < m - 1)
                    topRight[i][j] = Math.min(topRight[i][j], topRight[i][j + 1]);
            }
        }

        // 3️⃣ Bottom Left
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j < m; j++) {

                bottomLeft[i][j] = a[i][j];

                if (i < n - 1)
                    bottomLeft[i][j] = Math.min(bottomLeft[i][j], bottomLeft[i + 1][j]);

                if (j > 0)
                    bottomLeft[i][j] = Math.min(bottomLeft[i][j], bottomLeft[i][j - 1]);
            }
        }

        // 4️⃣ Bottom Right
        for (int i = n - 1; i >= 0; i--) {
            for (int j = m - 1; j >= 0; j--) {

                bottomRight[i][j] = a[i][j];

                if (i < n - 1)
                    bottomRight[i][j] = Math.min(bottomRight[i][j], bottomRight[i + 1][j]);

                if (j < m - 1)
                    bottomRight[i][j] = Math.min(bottomRight[i][j], bottomRight[i][j + 1]);
            }
        }

        int[] ans = new int[q];

        for (int k = 0; k < q; k++) {

            int R = queries[k][0] - 1;
            int C = queries[k][1] - 1;

            int sum = 0;

            // Top Left
            if (R > 0 && C > 0)
                sum += topLeft[R - 1][C - 1];

            // Top Right
            if (R > 0 && C < m - 1)
                sum += topRight[R - 1][C + 1];

            // Bottom Left
            if (R < n - 1 && C > 0)
                sum += bottomLeft[R + 1][C - 1];

            // Bottom Right
            if (R < n - 1 && C < m - 1)
                sum += bottomRight[R + 1][C + 1];

            ans[k] = sum;
        }

        return ans;
    }
}
```
### Question 3
Given an array arr[], find the minimum number of operations required to make the sum of its elements less than or equal to half of the original sum. In one operation, you may replace any element with half of its value (with floating-point precision).

Examples:

Input: arr[] = [8, 6, 2]
Output: 3
Explanation: Initial sum = (8 + 6 + 2) = 16, half = 8
Halve 8 → arr[] = [4, 6, 2], sum = 12 (still 12 > 8)
Halve 6 → arr[] = [4, 3, 2], sum = 9 (still 9 > 8)
Halve 2 → arr[] = [4, 3, 1], sum = 8. 

Input: arr[] = [9, 1, 2]
Output: 2
Explanation: Initial sum = 12, half = 6
Halve 9 → arr[] = [4.5, 1, 2], sum = 7.5 (still > 6)
Halve 4.5 → arr[] = [2.25, 1, 2], sum = 5.25 ≤ 6

Constraints:
1 ≤ arr.size() ≤ 105
0 ≤ arr[i] ≤ 104

```java
import java.util.*;

class Solution {
    public int minOperations(int[] arr) {

        PriorityQueue<Double> maxHeap =
            new PriorityQueue<>(Collections.reverseOrder());

        double sum = 0;

        for (int num : arr) {
            sum += num;
            maxHeap.add((double) num);
        }

        double target = sum / 2;
        int operations = 0;

        while (sum > target) {

            double largest = maxHeap.poll();
            double half = largest / 2;

            sum -= half;   // reduction amount

            maxHeap.add(half);
            operations++;
        }

        return operations;
    }
}
```
### Question 4
 You are given an array arr[] of size n , where arr[i] denotes the range of working hours a person at position i can cover.

    If arr[i] ≠ -1, the person at index i can work and cover the time interval [i - arr[i], i + arr[i]].
    If arr[i] = -1, the person is unavailable and cannot cover any time.

The task is to find the minimum number of people required to cover the entire working day from 0 to n - 1. If it is not possible to fully cover the day, return -1.

Examples:

Input: arr[] = [1, 2, 1, 0]
Output: 1
Explanation: The person at index 1 can cover the interval [-1, 3]. After adjusting to valid bounds, this becomes [0, 3], which fully covers the entire working day 0 to n -1. Therefore, only 1 person is required to cover the whole day.

Input: arr[] = [2, 3, 4, -1, 2, 0, 0, -1, 0]
Output: -1
Explanation: Persons up to index 2 cover interval [0…6], but working hour 7 cannot be cover as arr[7] = -1, Since the 7th hour cannot be covered by any person, it is impossible to cover the full working day.

Input: arr[] = [0, 1, 0, -1]
Output: -1
Explanation: The last hour cannot be covered by any person, so it is impossible to cover the full working day.

Constraints:
1 ≤ n ≤105
-1 ≤ arr[i] ≤ n

```java
class Solution {

    public int minMen(int[] arr) {

        int n = arr.length;
        int[] maxReach = new int[n];

        // Build max reach array
        for (int i = 0; i < n; i++) {

            if (arr[i] == -1)
                continue;

            int start = Math.max(0, i - arr[i]);
            int end = Math.min(n - 1, i + arr[i]);

            maxReach[start] = Math.max(maxReach[start], end);
        }

        int count = 0;
        int currentEnd = 0;
        int farthest = 0;

        for (int i = 0; i < n; i++) {

            farthest = Math.max(farthest, maxReach[i]);

            // If we cannot move forward
            if (i > farthest)
                return -1;

            // When we finish current coverage
            if (i == currentEnd) {

                // If already covering full range
                if (currentEnd >= n - 1)
                    return count;

                count++;
                currentEnd = farthest;
            }
        }

        return currentEnd >= n - 1 ? count : -1;
    }
}
```
### Question 5
You are given an integer array arr[ ]. For every element in the array, your task is to determine its Previous Smaller Element (PSE).

The Previous Smaller Element (PSE) of an element x is the first element that appears to the left of x in the array and is strictly smaller than x.

Note: If no such element exists, assign -1 as the PSE for that position.

Examples:

Input: arr[] = [1, 6, 2]
Output: [-1, 1, 1]
Explanation:
For 1, there is no element on the left, so answer is -1.
For 6, previous smaller element is 1.
For 2, previous smaller element is 1.

Input: arr[] = [1, 5, 0, 3, 4, 5]
Output: [-1, 1, -1, 0, 3, 4]
Explanation:
For 1, no element on the left, so answer is -1.
For 5, previous smaller element is 1.
For 0, no element on the left smaller than 0, so answer is -1.
For 3, previous smaller element is 0.
For 4, previous smaller element is 3.
For 5, previous smaller element is 4.

Constraints:
1 ≤ arr.size() ≤ 105
1 ≤ arr[i] ≤ 105

```java
import java.util.*;

class Solution {
    
    public static ArrayList<Integer> prevSmaller(int[] arr) {
        
        int n = arr.length;
        ArrayList<Integer> result = new ArrayList<>();
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < n; i++) {
            
            // Remove elements greater or equal to current
            while (!stack.isEmpty() && stack.peek() >= arr[i]) {
                stack.pop();
            }
            
            // If stack empty → no smaller element
            if (stack.isEmpty()) {
                result.add(-1);
            } else {
                result.add(stack.peek());
            }
            
            // Push current element
            stack.push(arr[i]);
        }
        
        return result;
    }
}
```
### Question 6
You are given an integer array arr[ ]. For every element in the array, your task is to determine its Previous Greater Element (PGE).

The Previous Greater Element (PGE) of an element x is the first element that appears to the left of x in the array and is strictly greater than x.

Note: If no such element exists, assign -1 as the PGE for that position.

Examples:

Input: arr[] = [10, 4, 2, 20, 40, 12, 30]
Output: [-1, 10, 4, -1, -1, 40, 40]
Explanation:
For 10, no elements on the left, so answer is -1.
For 4, previous greater element is 10.
For 2, previous greater element is 4.
For 20, no element on the left greater than 20, so answer is -1.
For 40, no element on the left greater than 40, so answer is -1.
For 12, previous greater element is 40.
For 30, previous greater element is 40.

Input: arr[] = [10, 20, 30, 40]
Output: [-1, -1, -1, -1]
Explanation: Each element of the array has no previous greater element.

Constraints:
1 ≤ arr.size() ≤ 105
1 ≤ arr[i] ≤ 105

```java
import java.util.*;

class Solution {
    
    public static ArrayList<Integer> preGreaterEle(int[] arr) {
        
        int n = arr.length;
        ArrayList<Integer> result = new ArrayList<>();
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < n; i++) {
            
            // Remove elements smaller or equal
            while (!stack.isEmpty() && stack.peek() <= arr[i]) {
                stack.pop();
            }
            
            // If stack empty → no greater element
            if (stack.isEmpty()) {
                result.add(-1);
            } else {
                result.add(stack.peek());
            }
            
            // Push current element
            stack.push(arr[i]);
        }
        
        return result;
    }
}
```
### Question 7
You are given an integer array arr[ ]. Your task is to count the number of subarrays where the first element is the minimum element of that subarray.

Note: A subarray is valid if its first element is not greater than any other element in that subarray.

Examples:

Input: arr[] = [1, 2, 1]
Output: 5
Explanation:
All possible subarrays are:
[1], [1, 2], [1, 2, 1], [2], [2, 1], [1]
Valid subarrays are:
[1], [1, 2], [1, 2, 1], [2], [1] -> total 5

Input: arr[] = [1, 3, 5, 2]
Output: 8
Explanation:
Valid subarrays are: [1], [1, 3], [1, 3, 5], [1, 3, 5, 2], [3], [3, 5], [5], [2] -> total 8

Constraints:
1 ≤ arr.size() ≤ 4*104
1 ≤ arr[i] ≤ 105

```java
import java.util.*;

class Solution {
    
    public int countSubarrays(int[] arr) {
        
        int n = arr.length;
        int[] nextSmaller = new int[n];
        Stack<Integer> stack = new Stack<>();
        
        // Step 1: Find Next Smaller Element index
        for (int i = n - 1; i >= 0; i--) {
            
            while (!stack.isEmpty() && arr[stack.peek()] >= arr[i]) {
                stack.pop();
            }
            
            if (stack.isEmpty())
                nextSmaller[i] = n;
            else
                nextSmaller[i] = stack.peek();
            
            stack.push(i);
        }
        
        // Step 2: Count subarrays
        int total = 0;
        
        for (int i = 0; i < n; i++) {
            total += nextSmaller[i] - i;
        }
        
        return total;
    }
}
```
### Question 8
You are given an array arr[]. The task is to determine whether the array contains a 132 pattern, i.e., three indices i,  j and k such that i < j < k , arr[i] < arr[j] > arr[k] and arr[i] < arr[k].
Return true if such a triplet exists, otherwise return false.

Examples:

Input: arr[] = [4, 7, 11, 5, 13, 2]
Output: true
Explanation: Triplet [4, 7, 5] satisfies the condition since 4 < 7, 5 < 7 and 4 < 5.

Input: arr[] = [11, 11, 12, 9]
Output: false
Explanation: No triplet satisfies the given conditions.

Constraints:
3 ≤ arr.size( ) ≤ 105
1 ≤ arr[i] ≤ 105

```java
import java.util.*;

class Solution {
    
    public boolean has132Pattern(int[] arr) {
        
        int n = arr.length;
        Stack<Integer> stack = new Stack<>();
        
        int second = Integer.MIN_VALUE;  // candidate for "2" in 132
        
        // Traverse from right to left
        for (int i = n - 1; i >= 0; i--) {
            
            // If current element is smaller than middle value
            if (arr[i] < second)
                return true;
            
            // Maintain decreasing stack
            while (!stack.isEmpty() && arr[i] > stack.peek()) {
                second = stack.pop();
            }
            
            stack.push(arr[i]);
        }
        
        return false;
    }
}
```
### Question 9
You are given an array arr[ ], where arr[i] represents the height of the ith person standing in a line.
A person i can see another person j if:

    height[j] < height[i],
    There is no person k standing between them such that height[k] ≥ height[i].

Each person can see in both directions (front and back).
Your task is to find the maximum number of people that any person can see (including themselves).

Examples:

Input: arr[] = [6, 2, 5, 4, 5, 1, 6 ]
Output: 6
Explanation:
Person 1 (height = 6) can see five other people at following positions (2, 3, 4, 5. 6) in addition to himself, i.e. total 6.
Person 2 (height: 2) can see only himself.
Person 3 (height = 5) is able to see people 2nd, 3rd, and 4th person.
Person 4 (height = 4) can see himself.
Person 5 (height = 5) can see people 4th, 5th, and 6th.
Person 6 (height =1) can only see himself.
Person 7 (height = 6) can see 2nd, 3rd, 4th, 5th, 6th, and 7th people.
A maximum of six people can be seen by Person 1, 7th

Input: arr[] = [1, 3, 6, 4]
Output: 4
Explanation: 
Person with height 6 can see persons with heights 1, 3 on the left and 4 on the right, along with himself, giving a total of 4.

Constraints:
1 ≤ arr.size() ≤ 104
1 ≤ arr[i] ≤ 105

```java
import java.util.*;

class Solution {
    
    public int maxPeople(int[] arr) {
        
        int n = arr.length;
        int[] prevGreater = new int[n];
        int[] nextGreater = new int[n];
        
        Stack<Integer> stack = new Stack<>();
        
        // Previous Greater or Equal
        for (int i = 0; i < n; i++) {
            
            while (!stack.isEmpty() && arr[stack.peek()] < arr[i]) {
                stack.pop();
            }
            
            if (stack.isEmpty())
                prevGreater[i] = -1;
            else
                prevGreater[i] = stack.peek();
            
            stack.push(i);
        }
        
        stack.clear();
        
        // Next Greater or Equal
        for (int i = n - 1; i >= 0; i--) {
            
            while (!stack.isEmpty() && arr[stack.peek()] < arr[i]) {
                stack.pop();
            }
            
            if (stack.isEmpty())
                nextGreater[i] = n;
            else
                nextGreater[i] = stack.peek();
            
            stack.push(i);
        }
        
        int maxSeen = 0;
        
        for (int i = 0; i < n; i++) {
            
            int left = i - prevGreater[i];
            int right = nextGreater[i] - i;
            
            int total = left + right - 1;
            
            maxSeen = Math.max(maxSeen, total);
        }
        
        return maxSeen;
    }
}
```
### Question 10
Given two integers n and k, the task is to find all valid combinations of k numbers that adds up to n based on the following conditions:

    Only numbers from the range [1, 9] used.
    Each number can only be used at most once.

Note: You can return the combinations in any order, the driver code will print them in sorted order.

Examples:

Input: n = 9, k = 3
Output: [[1, 2, 6], [1, 3, 5], [2, 3, 4]]
Explanation: There are three valid combinations of 3 numbers that sum to 9: [1 ,2, 6], [1, 3, 5] and [2, 3, 4].

Input: n = 3, k = 3
Output: []
Explanation: It is not possible to pick 3 distinct numbers from 1 to 9 that sum to 3, so no valid combinations exist.

Constraints:
1 ≤ n ≤ 50
1 ≤ k ≤ 9

```java
import java.util.*;

class Solution {
    
    public ArrayList<ArrayList<Integer>> combinationSum(int n, int k) {
        
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        backtrack(n, k, 1, new ArrayList<>(), result);
        return result;
    }
    
    private void backtrack(int target, int k, int start,
                           ArrayList<Integer> current,
                           ArrayList<ArrayList<Integer>> result) {
        
        // If valid combination found
        if (target == 0 && k == 0) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        // Stop if invalid
        if (target < 0 || k == 0)
            return;
        
        for (int i = start; i <= 9; i++) {
            
            current.add(i);
            
            backtrack(target - i, k - 1, i + 1, current, result);
            
            current.remove(current.size() - 1); // backtrack
        }
    }
}
```
### Question 11
Given an array arr[] of size n, your task is to divide the array in two subsets such that the absolute difference between the sum of elements in the two subsets is equal to zero (i.e., both subsets have the same sum).

    If n is even, both subsets must contain exactly n/2 elements.
    If n is odd, one subset must contain (n-1)/2 elements and the other subset must contain (n+1)/2 elements.

Note : If multiple answers exist, you may return any of them. The driver code will check and print true if your partition is valid, otherwise false.
It is guaranteed that there will always be atleast one valid partition.

Examples:

Input: arr[] = [1, 2, 3, 4]
Output: [[1, 4], [2, 3]]
Explanation: The absolute difference between the sum of both subsets is 0.

Input: arr[] = [5, 10, 15]
Output: [[5, 10], [15]]
Explanation: The absolute difference between the sum of both subsets is 0.

Constraints: 
1 ≤ n ≤ 20
-200 ≤ arr[i] ≤ 200

```java
import java.util.*;

class Solution {
    
    ArrayList<Integer> chosenIndices = new ArrayList<>();
    boolean found = false;
    
    public ArrayList<ArrayList<Integer>> equalPartition(int[] arr) {
        
        int n = arr.length;
        int totalSum = 0;
        
        for (int num : arr)
            totalSum += num;
        
        int k = n / 2;               // size constraint
        int target = totalSum / 2;   // required sum
        
        backtrack(arr, 0, k, 0, target, new ArrayList<>());
        
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        
        ArrayList<Integer> subset1 = new ArrayList<>();
        ArrayList<Integer> subset2 = new ArrayList<>();
        
        boolean[] used = new boolean[n];
        
        for (int idx : chosenIndices) {
            used[idx] = true;
        }
        
        for (int i = 0; i < n; i++) {
            if (used[i])
                subset1.add(arr[i]);
            else
                subset2.add(arr[i]);
        }
        
        result.add(subset1);
        result.add(subset2);
        
        return result;
    }
    
    private void backtrack(int[] arr, int index, int k,
                           int currentSum, int target,
                           ArrayList<Integer> currentIndices) {
        
        if (found) return;
        
        if (currentIndices.size() == k) {
            if (currentSum == target) {
                chosenIndices = new ArrayList<>(currentIndices);
                found = true;
            }
            return;
        }
        
        if (index >= arr.length)
            return;
        
        // Choose element
        currentIndices.add(index);
        backtrack(arr, index + 1, k,
                  currentSum + arr[index], target,
                  currentIndices);
        currentIndices.remove(currentIndices.size() - 1);
        
        // Skip element
        backtrack(arr, index + 1, k,
                  currentSum, target,
                  currentIndices);
    }
}
```
### Question 12
Given an array arr[] of integers and an integer k, select k elements from the array such that the minimum absolute difference between any two of the selected elements is maximized. Return this maximum possible minimum difference.

Examples:

Input: arr[] = [2, 6, 2, 5], k = 3
Output: 1
Explanation: 3 elements out of 4 elements are to be selected with a minimum difference as large as possible. Selecting 2, 2, 5 will result in minimum difference as 0. Selecting 2, 5, 6 will result in minimum difference as 6 - 5 = 1.

Input: arr[] = [1, 4, 9, 0, 2, 13, 3], k = 4
Output: 4
Explanation: Selecting 0, 4, 9, 13 will result in minimum difference of 4, which is the largest minimum difference possible.

Constraints:
1 ≤ arr.size() ≤ 105
0 ≤ arr[i] ≤ 106
2 ≤ k ≤ arr.size() 

```java
import java.util.*;

class Solution {
    
    public int maxMinDiff(int[] arr, int k) {
        
        Arrays.sort(arr);
        
        int low = 0;
        int high = arr[arr.length - 1] - arr[0];
        int answer = 0;
        
        while (low <= high) {
            
            int mid = low + (high - low) / 2;
            
            if (canPlace(arr, k, mid)) {
                answer = mid;     // possible, try bigger
                low = mid + 1;
            } else {
                high = mid - 1;   // not possible
            }
        }
        
        return answer;
    }
    
    private boolean canPlace(int[] arr, int k, int minDiff) {
        
        int count = 1;  // first element always chosen
        int last = arr[0];
        
        for (int i = 1; i < arr.length; i++) {
            
            if (arr[i] - last >= minDiff) {
                count++;
                last = arr[i];
            }
            
            if (count >= k)
                return true;
        }
        
        return false;
    }
}
```
### Question 13
Given a sorted array arr[] containing distinct non negative integers that has been rotated at some unknown pivot, and a value x. Your task is to count the number of elements in the array that are less than or equal to x.

Examples:

Input: arr[] = [4, 5, 8, 1, 3], x = 6
Output: 4
Explanation: 1, 3, 4 and 5 are less than 6, so the count of all elements less than 6 is 4.

Input: arr[] = [6, 10, 12, 15, 2, 4, 5], x = 14
Output: 6
Explanation: All elements except 15 are less than 14, so the count of all elements less than 14 is 6.

Constraints:
1 ≤ arr.size() ≤ 105
0 ≤ arr[i], x ≤ 109

```java
import java.util.*;

class Solution {
    
    public int countLessEqual(int[] arr, int x) {
        
        int n = arr.length;
        
        // Step 1: Find pivot (smallest element)
        int low = 0, high = n - 1;
        
        while (low < high) {
            int mid = low + (high - low) / 2;
            
            if (arr[mid] > arr[high])
                low = mid + 1;
            else
                high = mid;
        }
        
        int pivot = low;
        
        // Step 2: Count in two sorted parts
        int count = 0;
        
        count += upperBound(arr, pivot, n - 1, x);
        count += upperBound(arr, 0, pivot - 1, x);
        
        return count;
    }
    
    // returns count of elements ≤ x in arr[l..r]
    private int upperBound(int[] arr, int l, int r, int x) {
        
        if (l > r) return 0;
        
        int low = l, high = r;
        
        while (low <= high) {
            int mid = low + (high - low) / 2;
            
            if (arr[mid] <= x)
                low = mid + 1;
            else
                high = mid - 1;
        }
        
        return low - l;
    }
}
```
### Question 14
You will be given a vector of integers of size n containing the elements. Your task is to find the sum of all the integers present in the array.As the sum can be large you have to return a value in long long data type.

Example 1:

Input:
n=7
input= {6,2,5,4,5,1,6}

Output : 29

Constraints:
1 ≤  n ≤ 105
1 ≤ arr[i] ≤ 105 

Expected Time Complexity: O(n)

Expected Space Complexity: O(1)

```java
class Solution{
    public:
    long long get_Sum(int n, vector<int>& input)
    {
        long long sum = 0;   // use long long
        
        for(int i = 0; i < n; i++)
        {
            sum += input[i];
        }
        
        return sum;
    }
};
```