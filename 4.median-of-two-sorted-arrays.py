#
# @lc app=leetcode id=4 lang=python3
#
# [4] Median of Two Sorted Arrays
#
from typing import List
import pytest


# @lc code=start
import math


class Solution:
    def findMedianSortedArrays(
        self, nums1: List[int], nums2: List[int]
    ) -> float | None:
        """
        Finds the median value of a sorted array, where that array is the
        combination of two sorted arrays and sorted is defined as smallest
        to largest.

        The algorithm's ultimate goal is to find the value at the median index
        (or indices when `m + n` is even, where `m` is the length of `nums1` and
        `n` is the length of `nums2`; a.k.a. the index of interest) of a combined
        and sorted array without combining and sorting the array.

        It starts by picking a median number of the array with the most amount of
        elements. This will reduce the longest array by a factor of ~2 on every
        iteration.

        For each number picked, the algorithm finds the index of that number in
        the combined and sorted array (using binary search under the hood).

        If the index found is less than the index of interest, the next iteration
        of the algorithm will only operate on the side of each list closer to the
        median index, dropping ~1/2 of the longer list and some amount of the
        shorter list.

        Once the index of interest is found during the index search, it's value
        is returned.
        """
        if len(nums1) == 0 and len(nums2) == 0:
            return None

        # The total length of an array comprised of both `nums1` and `nums2`.
        combined_len = len(nums1) + len(nums2)

        # An indicator of whether to use the value at a given index or the
        # average value of two consecutive indices.
        is_even = combined_len % 2 == 0

        # The index of interest is the index of the median value in a combined
        # and sorted list. If the length of the combined list is even, the
        # indices of interest are the index of interest and the index to the
        # left.
        index_of_interest = combined_len // 2

        # The `nums1` index offset tracks the total numbers discarded to the left of the
        # `nums1` list. Thus, the actual index is the intermediate `nums1` index plus
        # the `nums1` index offset.
        nums1_index_offset = 0

        # The `nums2` index offset tracks the total numbers discarded to the left of the
        # `nums2` list. Thus, the actual index is the intermediate `nums2` index plus
        # the `nums2` index offset.
        nums2_index_offset = 0

        # The combined index offset tracks the total numbers discarded to the left of
        # each array in the uncomputed combined array. Thus, the actual combined index
        # is the intermediate combined index plus the combined index offset
        combined_index_offset = 0

        # Create references to the original lists for final median calculations.
        nums1_source = nums1
        nums2_source = nums2

        while True:
            # Operating on the longest list cuts down the problem size the
            # fastest.
            longest_list = nums1 if len(nums1) > len(nums2) else nums2

            # If the length of both arrays are the same, there is a significant
            # opportunity for the maintainer of this code to unlink `longest_list`
            # from the list operated on. This ID is a seatbelt.
            longest_list_id = "nums1" if len(nums1) > len(nums2) else "nums2"

            # Using a median index is crucial to dropping all values either
            # left or right of the median value.
            median_index = len(longest_list) // 2

            # The median value is chosen as a starting point as it is likely
            # closer to the combined median index.
            median_value = longest_list[median_index]

            # This intermediary combined index is the index of the supplied
            # value in the reduced lists of the current iteration. To find
            # the global combined index, add this intermediary combined index
            # to the index offset.
            combined_index = self._combined_index(nums1, nums2, median_value)

            # Check if the combined index matches the index of interest
            global_combined_index = combined_index + combined_index_offset
            if global_combined_index == index_of_interest or (
                is_even and global_combined_index == index_of_interest - 1
            ):
                # If the combined list has odd length, return the value
                if not is_even:
                    return median_value

                # If the combined list is even, find the additional value
                if global_combined_index == index_of_interest:
                    # The additional value to find has a global combined index one
                    # less than the index of interest. This means that the additional
                    # value can be found using: 1) the value one index left of the
                    # longest list's median index and 2) the value at the index
                    # returned by a binary search (rounded down if necessary) in the
                    # shorter list. The max value between the two will take the
                    # index position one left of the index of interest.

                    # Get the value immediately left of the median value in the longest list
                    longest_list_full = (
                        nums1_source if longest_list_id == "nums1" else nums2_source
                    )
                    longest_list_left_index = (
                        (
                            nums1_index_offset
                            if longest_list_id == "nums1"
                            else nums2_index_offset
                        )
                        + median_index
                        - 1
                    )
                    if 0 <= longest_list_left_index < len(longest_list_full):
                        longest_list_left_value = longest_list_full[
                            longest_list_left_index
                        ]
                    else:
                        longest_list_left_value = -math.inf

                    # Get the value immediately left of the median value in the shortest list.
                    # If the index returned is a half value, subtracting by 0.5 gives the
                    # index immediately left. If the index returned is a whole value,
                    # subtracting by 0.5 then rounding will return the index immediately left
                    # as well.
                    shortest_list_full = (
                        nums1_source if longest_list_id != "nums1" else nums2_source
                    )
                    shortest_list_left_index = round(
                        self._binary_search(shortest_list_full, median_value) - 0.5
                    )

                    # The binary search possibly returns an index outside of the list, when
                    # the element being searched for would be found before the start of the
                    # list or after the end of the list.

                    # If the index returned is outside of the list on the right, the
                    # shortest list's index left of the median value would be the returned
                    # index minus one.
                    if shortest_list_left_index == len(shortest_list_full):
                        shortest_list_left_index -= 1

                    # If the index returned is outside of the list on the left, the
                    # shortest list's index left of the median value does not exist.
                    # Thus, the value in the shortest list left of the median index
                    # should be set to negative infinity.
                    if 0 <= shortest_list_left_index < len(shortest_list_full):
                        shortest_list_left_value = shortest_list_full[
                            shortest_list_left_index
                        ]

                    else:
                        shortest_list_left_value = -math.inf

                    # Even though either of the left values COULD be negative infinity,
                    # it's impossible for BOTH to be negative infinity, per the constraints.
                    left_value = max(longest_list_left_value, shortest_list_left_value)

                    return (left_value + median_value) / 2

                else:  # global_combined_index == index_of_interest - 1
                    # The additional value to find has a global combined index one
                    # greater than the index of interest

                    # Get the value immediately right of the median value in the longest list
                    longest_list_full = (
                        nums1_source if longest_list_id == "nums1" else nums2_source
                    )
                    longest_list_right_index = (
                        (
                            nums1_index_offset
                            if longest_list_id == "nums1"
                            else nums2_index_offset
                        )
                        + median_index
                        + 1
                    )
                    if longest_list_right_index < len(longest_list_full):
                        longest_list_right_value = longest_list_full[
                            longest_list_right_index
                        ]
                    else:
                        longest_list_right_value = math.inf

                    # Get the value immediately right of the median value in the shortest list.
                    # If the index returned is a half value, rounding gives the index immediately
                    # right of the median value. If the index returned is a whole value, rounding
                    # doesn't affect the index.
                    shortest_list_full = (
                        nums1_source if longest_list_id != "nums1" else nums2_source
                    )
                    shortest_list_right_index = round(
                        self._binary_search(shortest_list_full, median_value)
                    )

                    # The binary search possibly returns an index outside of the list, when
                    # the element being searched for would be found before the start of the
                    # list or after the end of the list.

                    # If the index returned is outside of the list on the left, the
                    # shortest list's index right of the median value would be the returned
                    # index plus one.
                    if shortest_list_right_index == -1:
                        shortest_list_right_index += 1

                    # If the index returned is outside of the list on the right, the
                    # shortest list's index right of the median value does not exist.
                    # Thus, the value in the shortest list right of the median index
                    # should be set to negative infinity.
                    if 0 <= shortest_list_right_index < len(shortest_list_full):
                        shortest_list_right_value = shortest_list_full[
                            shortest_list_right_index
                        ]

                    else:
                        shortest_list_right_value = math.inf

                    right_value = min(
                        longest_list_right_value, shortest_list_right_value
                    )

                    return (median_value + right_value) / 2

            # If the global combined index is less than the index of interest,
            # then every value less than or equal to the index of the median
            # value chosen should be dropped from the window of interest.
            elif global_combined_index < index_of_interest:
                # Shrink the left side of the window of interest from the
                # longest list.
                if longest_list_id == "nums1":
                    nums1 = nums1[median_index + 1 :]
                    nums1_index_offset += median_index + 1
                else:
                    nums2 = nums2[median_index + 1 :]
                    nums2_index_offset += median_index + 1

                combined_index_offset += median_index + 1

            # If the global combined index is greater than the index of
            # interest, then every value greater than or equal to the index
            # of the median value chosen should be dropped from the window of
            # interest.
            else:  # global_combined_index > index_of_interest
                # Shrink the right side of the window of interest from the
                # longest list.
                if longest_list_id == "nums1":
                    nums1 = nums1[:median_index]
                else:
                    nums2 = nums2[:median_index]

    def _binary_search(self, nums: List[int], value: int) -> float:
        """
        Uses binary search to identify the index of `value` in `nums`.

        If `value` is in `nums`, the leftmost index of `value` is returned. If
        `value` is not in `nums`, a float index is returned, such that the
        float is half way between the index left and right of `value`.

        For instance, suppose `nums` is `[1, 3]` and `value` is `2`. The returned
        value would be `0.5`, half way between the index of the largest value
        less than `2` and the smallest value greater than `2`.

        ASSUMPTION: `nums` is sorted smallest to largest.
        """
        if len(nums) == 0:
            return 0

        if value < nums[0]:
            return -1

        search_index = len(nums) // 2

        # If the current search index is the location of the value being
        # searched for, return it.
        if value == nums[search_index]:
            return search_index

        # If the search value is between the current index and its previous
        # index, return the half index to the left.
        elif search_index > 0 and nums[search_index - 1] < value < nums[search_index]:
            return search_index - 0.5

        # If the search value is between the current index and its next
        # index, return the half index to the right.
        elif (
            search_index < len(nums) - 1
            and nums[search_index] < value < nums[search_index + 1]
        ):
            return search_index + 0.5

        # Otherwise, update the search window to the side containing the
        # search value.
        if value < nums[search_index]:
            return self._binary_search(nums[:search_index], value)

        else:  # value > nums[search_index]:
            return (
                self._binary_search(nums[search_index + 1 :], value) + search_index + 1
            )

    def _combined_index(self, nums1: List[int], nums2: List[int], value: int) -> int:
        """
        Determines the index of `value` if `nums1` and `nums2` were combined and sorted.

        If `value` is not in the inclusive range [min(`nums1` + `nums2`), max(`nums1` + `nums2`)],
        a combined index returned will not be in the range [0, `len(nums1) + len(nums2)`).
        Specifically, the returned value would be in the set {-1, `len(nums1) + len(nums2)`}.

        ASSUMPTION: `nums1` and `nums2` are sorted smallest to largest.
        """
        # Find the index of `value` in `nums1`. If the index returned includes
        # a half, round it up. This equates to the integer index to the left
        # plus one, where the added one accounts for 0-indexing.
        nums1_index = round(self._binary_search(nums1, value))

        # Do the same for `nums2`.
        nums2_index = round(self._binary_search(nums2, value))

        if nums1_index < 0 and nums2_index < 0:
            return -1

        elif nums1_index < 0 or nums2_index < 0:
            return max(0, nums1_index) + max(0, nums2_index)

        else:
            return nums1_index + nums2_index


# @lc code=end

if __name__ == "__main__":
    s = Solution()

    # Testing _binary_search
    assert s._binary_search([1, 2], 0) == -1
    assert s._binary_search([1, 2], 1) == 0
    assert s._binary_search([1, 3], 2) == 0.5
    assert s._binary_search([1, 2], 2) == 1
    assert s._binary_search([1, 2], 3) == 2

    # Testing _combined_index
    assert s._combined_index([1], [2], 0) == -1
    assert s._combined_index([1], [2], 1) == 0
    assert s._combined_index([1], [3], 2) == 1
    assert s._combined_index([1], [2], 2) == 1
    assert s._combined_index([1], [2], 3) == 2

    # Testing findMedianSortedArrays
    assert s.findMedianSortedArrays([1, 3], [2]) == 2
    assert s.findMedianSortedArrays([1], [2]) == 1.5
    assert s.findMedianSortedArrays([1, 2], [3, 4]) == 2.5
    assert s.findMedianSortedArrays([], []) == None
    assert s.findMedianSortedArrays([], [1]) == 1.0
    assert s.findMedianSortedArrays([], [1, 2]) == 1.5
    assert s.findMedianSortedArrays([1], [2]) == 1.5
    assert s.findMedianSortedArrays([1, 2, 3, 4, 5, 6], [100]) == 4
    assert s.findMedianSortedArrays([1, 2, 3], [10, 11, 12]) == 6.5
    assert s.findMedianSortedArrays([1, 3, 5], [2, 4, 6]) == 3.5
    assert s.findMedianSortedArrays([5, 5, 5], [5, 5, 5]) == 5.0
    assert s.findMedianSortedArrays([-5, -3, -1], [1, 3, 5]) == 0.0
    assert s.findMedianSortedArrays([1, 4, 7], [2, 3, 6, 8]) == 4.0
    assert s.findMedianSortedArrays([1], [2, 3, 4, 5, 6, 7, 8, 9]) == 5.0
    assert s.findMedianSortedArrays([1, 2, 2, 2], [2, 2, 3, 4]) == 2.0
    assert s.findMedianSortedArrays([1, 2, 2, 2], [2, 2, 3]) == 2.0
    assert s.findMedianSortedArrays([2, 2, 4, 4], [2, 2, 4, 4]) == 3.0
    assert s.findMedianSortedArrays([1, 2, 3, 4, 5], [100]) == 3.5
    assert s.findMedianSortedArrays([1, 2, 3, 4], [5, 6, 7, 8]) == 4.5
    assert s.findMedianSortedArrays([1, 2, 2, 2, 3], [2, 2, 2, 4, 5]) == 2.0
    assert s.findMedianSortedArrays([-3, -2, -1], [1, 2, 3, 4]) == 1.0
    assert s.findMedianSortedArrays([1], [100]) == 50.5
    assert s.findMedianSortedArrays([1, 1, 1, 1, 1], [1000, 1001, 1002]) == 1.0
    assert s.findMedianSortedArrays([1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]) == 6.5
