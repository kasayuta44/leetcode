from collections import deque
import heapq
from pprint import pprint


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isPalindrome(s):
    s.lower()
    chars = []
    for c in s:
        if c.isalpha():
            chars.append(c)
    for i in range(len(chars)//2):
        if not chars[i] == chars[-1 * i - 1]:
            return False
    return True

def maxProfit(prices):
    maxDay = {}
    for i, buy in enumerate(prices):
        for j, sell in enumerate(prices):
            if i < j:
                if sell - buy > 0:
                    try:
                        if maxDay[i] < sell - buy:
                            maxDay[i] = sell - buy
                    except KeyError:
                        maxDay[i] = sell - buy
                else:
                    maxDay[i] = 0
    return max(maxDay.values())

def removeElement(nums, val):
    if not nums:
        return 0
    if val > 50:
        return len(nums)
    k = 0
    lbound = 0
    for rbound in range(len(nums)):
        if nums[rbound] != val:
            nums[lbound], nums[rbound] = nums[rbound], nums[lbound]
            lbound += 1
            k += 1
    
    return k, nums

class MyLinkedList:

    class ListNode:
        def __init__(self, val, next = None, prev = None):
            self.val = val
            self.next = next
            self.prev = prev
        
    def __init__(self):
        self.head = None
        self.tail = None

    def get(self, index: int) -> int:
        curr = self.head
        while index > 0 and curr:
            curr = curr.next
            index -= 1
        if curr:
            return curr.val
        else:
            return -1

    def addAtHead(self, val: int) -> None:
        if self.head:
            newHead = self.ListNode(val)
            newHead.next = self.head
            self.head.prev = newHead
            self.head = newHead
        else:
            self.head = self.ListNode(val)
            self.tail = self.head

    def addAtTail(self, val: int) -> None:
        if self.tail:
            newTail = self.ListNode(val)
            newTail.prev = self.tail
            self.tail.next = newTail
            self.tail = newTail
        else:
            self.tail = self.ListNode(val)
            self.head = self.tail

    def addAtIndex(self, index: int, val: int) -> None:
        #new node goes between curr and prev such that prev -> new node -> curr
        curr = self.head
        prev = None
        while index > 0 and curr:
            prev = curr
            curr = curr.next
            index -= 1
            print("add", curr.val, index)
        if not prev and index == 0:
            self.addAtHead(val)
        elif not curr and index == 0:
            self.addAtTail(val)
        elif index == 0 and curr and prev:
            newNode = self.ListNode(val)
            newNode.next = curr
            newNode.prev = prev
            prev.next = newNode
            curr.prev = newNode

    def deleteAtIndex(self, index: int) -> None:
        curr = self.head
        while index > 0 and curr:
            curr = curr.next
            index -= 1
        if index == 0:
            if curr == self.head:
                if curr.next:
                    self.head = curr.next
                    curr.next.prev = None
                else:
                    self.head = None
            elif curr == self.tail:
                if curr.prev:
                    self.tail = curr.prev
                    curr.prev.next = None
                else:
                    self.tail = None
            elif curr:
                curr.prev.next = curr.next
                curr.next.prev = curr.prev
    
    def reverseList(self):
        prev = None
        curr = self.head
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        self.head = prev

    def printList(self):
        curr = self.head
        while curr:
            print(curr.val)
            curr = curr.next

class BrowserHistory:

    class pageNode:
        def __init__(self, val = ''):
            self.val = val
            self.next = None
            self.prev = None

    def __init__(self, homepage: str):
        node = self.pageNode(homepage)
        self.homepage = node
        self.currentpage = node

    def visit(self, url: str) -> None:
        visitNode = self.pageNode(url)
        self.currentpage.next = visitNode
        visitNode.prev = self.currentpage
        self.currentpage = visitNode
        print(f"visiting {self.currentpage.val}")
        print(f"previous {self.currentpage.prev.val}")

    def back(self, steps: int) -> str:
        while steps > 0:
            if self.currentpage.prev:
                self.currentpage = self.currentpage.prev
            else:
                break
            steps -= 1
        print(f"went back to {self.currentpage.val}")

    def forward(self, steps: int) -> str:
        while steps > 0:
            if self.currentpage.next:
                self.currentpage = self.currentpage.next
            else:
                break
        print(f"went forward to {self.currentpage.val}")

class MyStack:
    class q:
        class node:
            def __init__(self, val):
                self.val = val
                self.next = None

        def __init__(self):
            self.head = None
            self.tail = None
            self.size = 0
        def add(self, val):
            n = self.node(val)
            if self.tail:
                self.tail.next = n
                self.tail = n
            else:
                self.head = n
                self.tail = n
            self.size += 1
        def peek(self):
            return self.head
        def pop(self):
            temp = self.head
            if self.head.next:
                self.head = self.head.next
            else:
                self.head = None
                self.tail = None
            self.size -= 1
            return temp.val

        def getSize(self):
            return self.size

        def isEmpty(self):
            if self.size == 0: return True
            else: return False

    def __init__(self):
        self.q1 = self.q()
        self.q2 = self.q()

    def push(self, x: int) -> None:
        if self.q2.isEmpty():
            print("q2 is empty")
            self.q1.add(x)
        else:
            self.q2.add(x)

    def pop(self) -> int:
        if self.q1.isEmpty():
            for i in range(self.q2.getSize() - 1):
                self.q1.add(self.q2.pop())
            return self.q2.pop()
        else:
            for i in range(self.q1.getSize() - 1):
                self.q2.add(self.q1.pop())
            return self.q1.pop()

    def top(self) -> int:
        if self.q1.isEmpty():
            for i in range(self.q2.getSize() - 1):
                self.q1.add(self.q2.pop())
            temp = self.q2.peek()
            self.q1.add(self.q2.pop())
            return temp.val
        else:
            for i in range(self.q1.getSize() - 1):
                self.q2.add(self.q1.pop())
            temp = self.q1.peek()
            self.q2.add(self.q1.pop())
            return temp.val

    def empty(self) -> bool:
        return self.q1.isEmpty() and self.q2.isEmpty()

def search(nums, target):
    l = 0
    r = len(nums) - 1
    print(l, r)
    while l + 1 < r:
        mid = (r + l) // 2
        print(f"mid {mid} r {r} l {l}")
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            r = mid
        else:
            l = mid
    if nums[l] == target:
        return l
    elif nums[r] == target:
        return r
    return -1

def searchMatrix(matrix, target):
    top = 0
    bottom = len(matrix) - 1
    while top < bottom:
        middle = (top + bottom) // 2
        if matrix[middle][0] == target:
            return True
        elif matrix[middle][0] > target:
            bottom = middle - 1
        else:
            top = middle + 1

    left = 0
    right = len(matrix[0]) - 1
    
    if matrix[top][0] > target:
        top -= 1

    while left < right:
        middle = (left + right) // 2
        if matrix[top][middle] == target:
            return True
        elif matrix[top][middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    
    if matrix[top][left] == target:
        return True
    print(f"bottom {bottom}, top {top}, left {left}, right {right}")
    return False

def eat(k, piles, h):
    for banana in piles:
        if h < 0:
            break
        if banana % k > 0:
            s = banana // k + 1
        else:
            s = banana // k
        h -= s

    return h #when piles is finished

def minEatingSpeed(piles, h: int) -> int:
    tHigh = h // len(piles)
    tLow = h - len(piles) + 1
    low = max(piles) // tLow
    if low == 0:
        low = 1
    high = (max(piles) // tHigh) + (max(piles) % tHigh)
    while low != high:
        middle = (low + high) // 2
        print(f"low: {low}, middle: {middle}, high: {high}")
        eatResult = eat(middle, piles, h)
        print(eatResult)
        if eatResult < 0:
            low = middle + 1
        else:
            high = middle
    
    return high

def rightSideView(root):
    q = deque()
    res = []
    if root:
        q.append(root)
        res.append(root.val)
    while len(q) > 0:
        print('new while loop')
        for i in range(len(q)):
            print('new for loop')
            curr = q.popleft()
            rem = len(q)
            if curr.right:
                q.append(curr.right)
                res.append(curr.right.val)
                print(f'curr right: {curr.right.val}')
                if curr.left:
                    q.append(curr.left)
                    print(f'curr left: {curr.left.val}')
                for j in range(rem):
                    curr2 = q.popleft()
                    if curr2.right:
                        q.append(curr2.right)
                        print(f'curr2 right: {curr2.right.val}')
                    if curr2.left:
                        q.append(curr2.left)
                        print(f'curr2 left: {curr2.left.val}')
                break
            if curr.left:
                q.append(curr.left)
                res.append(curr.left.val)
                for j in range(rem):
                    curr2 = q.popleft()
                    if curr2.right:
                        q.append(curr2.right)
                    if curr2.left:
                        q.append(curr2.left)
                break
    return res

def combinationSum(candidates, target):
    res = []
    subset = []

    def dfs(i):
        print(i)
        print(subset, res)
        if i > len(candidates) - 1:
            return
        if sum(subset) == target:
            res.append(subset.copy())
        elif sum(subset) > target:
            return #under the assumption that candidates is a sorted list
        subset.append(candidates[i])
        dfs(i)
        dfs(i+1)
        subset.pop()
        dfs(i+1)
    
    dfs(0)
    return res

def jump(nums):
    high = 0
    for i in range(len(nums)):
        if high < i:
            return False
        if i + nums[i] > high:
            high = i + nums[i]
        if high >= len(nums) - 1:
            return True
        
def maxAreaOfIsland(grid):
    high = 0
    visited = set()
    rows, cols = len(grid), len(grid[0])
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def dfs(r, c, area):
        if (r not in range(rows) or
            c not in range(cols) or
            grid[r][c] == 0 or
            (r,c) in visited):
            return 0
        
        visited.add((r, c))
        area += 1
        for dr, dc in directions:
            area = dfs(r + dr, c + dc, area)

        return area

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in visited:
                high = max(high, dfs(r, c, 0))
    
    return high

def dfs(grid, r, c, visited):
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    rows, cols = len(grid), len(grid[0])
    if (r not in range(rows) or
        c not in range(cols) or
        grid[r][c] == 0 or
        (r,c) in visited):
        return 0
    visited.add((r, c))
    area = 1
    for dr, dc in directions:
        area += dfs(grid, r + dr, c + dc, visited)
    return area

def buildTree(preorder, inorder):
    print(preorder, inorder)
    root = TreeNode(preorder[0])
    if len(inorder) == 1:
        preorder.pop(0)
        return root

    i = inorder.index(preorder.pop(0))
    if inorder[:i]:
        root.left = buildTree(preorder, inorder[:i])
    if inorder[i + 1:]:
        root.right = buildTree(preorder, inorder[i + 1:])

    return root

def numOfSubarrays(arr, k, threshold):
    count = 0
    total = 0
    L = 0

    for R in range(len(arr)):
        print(L, R, total)
        total += arr[R]
        if R - L + 1 > k:
            total -= arr[L]
            L += 1
        if R - L + 1 == k and (total / k) >= threshold:
            count += 1

    return count

class SegmentTree:
    def __init__(self, total, L, R):
        self.sum = total
        self.left = None
        self.right = None
        self.L = L
        self.R = R

    @staticmethod
    def build(nums, L, R):
        if L == R:
            return SegmentTree(nums[L], L, R)
        M = (L + R) // 2
        root = SegmentTree(0, L, R)
        root.left = SegmentTree.build(nums, L, M)
        root.right = SegmentTree.build(nums, M + 1, R)
        root.sum = root.left.sum + root.right.sum
        return root
    
    def update(self, index, val):
        if self.L == self.R:
            self.sum = val
            return
        M = (self.L + self.R) //2
        if index > M:
            self.right.update(index, val)
        else:
            self.left.update(index, val)
        self.sum = self.left.sum + self.right.sum

    def rangeQuery(self, L, R):
        if self.L == L and self.R == R:
            return self.sum
        M = (self.L + self.R) // 2
        if L > M:
            return self.right.rangeQuery(self, L, R)
        elif R <= M:
            return self.left.rangeQuery(self, L, R)
        else:
            return (self.left.rangeQuery(L, M) + 
                    self.right.rangeQuery(M + 1, R))

class NumArray:

    def __init__(self, nums):
        self.root = SegmentTree.build(nums, 0, len(nums) - 1)

    def update(self, index, val):
        self.root.update(index, val)

    def sumRange(self, left, right):
        return self.root.rangeQuery(left, right)

MOD = 10**9+7
def numFactoredBinaryTrees(arr):
    arr.sort()
    d = {x:1 for x in arr}
    s = set(arr)

    for i in arr:
        for j in arr:
            if j > i**0.5:
                break
            if i % j == 0 and i // j in s:
                if i // j == j:
                    d[i] += d[j] * d[j]
                else:
                    d[i] += d[j] * d[i // j] * 2
                d[i] %= MOD
        pass
    return sum(d.values()) % MOD

def lengthOfLIS(nums):
    res = [1]
    def dfs(i, sub, prev):
        print(i, sub, prev)
        if  i >= len(nums):
            res[0] = max(res[0], len(sub))
            return
        sub.append(nums[i])
        dfs(i + 1, sub, nums[i])
        sub.pop()
        dfs(i + 1, sub, prev)

    dfs(0, [], float('-inf'))
    return res[0]

class TwoHeap:
    def __init__(self):
        self.small, self.large, self.length = [], [], 0

    def getLength(self):
        return self.length

    def check(self):
        while self.small and self.large and -1 * self.small[0] > self.large[0]:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        while len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        while len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)

    def add(self, n):
        print('adding: ', n)
        self.length += 1
        heapq.heappush(self.small, -1 * n)
        self.check()

    def remove(self, n):
        print('removing: ', n)
        self.length -= 1
        temp = []
        if n <= -1 * self.small[0]:
            val = -1 * heapq.heappop(self.small)
            while val != n:
                temp.append(val)
                val = -1 * heapq.heappop(self.small)
        else:
            val = heapq.heappop(self.large)
            while val != n:
                temp.append(val)
                val = heapq.heappop(self.large)
        for i in temp:
            self.add(i)
        self.check()

    def getMedian(self):
        print('getting median')
        if len(self.small) > len(self.large):
            print('odd, small has more values: ', -1 * self.small[0])
            return -1 * self.small[0]
        elif len(self.large) > len(self.small):
            print('odd, large has more values: ', self.large[0])
            return self.large[0]
        print('even values in each: ', ((-1 * self.small[0]) + self.large[0]) / 2)
        return ((-1 * self.small[0]) + self.large[0]) / 2

def medianSlidingWindow(nums, k):
    res = []
    h = TwoHeap()
    for i in range(len(nums)):
        h.add(nums[i])
        if h.getLength() > k:
            h.remove(nums[i - k])
        if i + 1 >= k:
            res.append(h.getMedian())
        print()
    return res

print(medianSlidingWindow([2147483647,1,2,3,4,5,6,7,2147483647], 2))