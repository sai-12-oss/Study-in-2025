class SplayTree:
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def is_empty(self):
        return self.root is None

    def insert(self, key):
        self.root = self._insert_rec(self.root, key)
        self.root = self._splay(self.root, key)

    def search(self, key):
        self.root = self._splay(self.root, key)
        return self.root is not None and self.root.key == key

    def delete(self, key):
        if self.is_empty():
            raise Exception("Cannot delete from an empty tree")

        self.root = self._splay(self.root, key)

        if self.root.key != key:
            return

        if self.root.left is None:
            self.root = self.root.right
        else:
            temp = self.root
            self.root = self._splay(self.root.left, self._find_max(self.root.left).key)
            self.root.right = temp.right

    def _find_max(self, root):
        while root.right is not None:
            root = root.right
        return root

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        y.right = x
        return y

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        y.left = x
        return y

    def _splay(self, root, key):
        if root is None or root.key == key:
            return root

        if root.key > key:
            if root.left is None:
                return root
            if root.left.key > key:
                root.left.left = self._splay(root.left.left, key)
                root = self._rotate_right(root)
            elif root.left.key < key:
                root.left.right = self._splay(root.left.right, key)
                if root.left.right is not None:
                    root.left = self._rotate_left(root.left)
            return root if root.left is None else self._rotate_right(root)
        else:
            if root.right is None:
                return root
            if root.right.key > key:
                root.right.left = self._splay(root.right.left, key)
                if root.right.left is not None:
                    root.right = self._rotate_right(root.right)
            elif root.right.key < key:
                root.right.right = self._splay(root.right.right, key)
                root = self._rotate_left(root)
            return root if root.right is None else self._rotate_left(root)

    def _insert_rec(self, root, key):
        if root is None:
            return self.Node(key)

        if key < root.key:
            root.left = self._insert_rec(root.left, key)
        elif key > root.key:
            root.right = self._insert_rec(root.right, key)
        else:
            raise Exception(f"Duplicate key: {key}")

        return root
def find(tree, key):
    """
    Find a key in the splay tree and splay it to the root.

    :param tree: The SplayTree instance.
    :param key: The key to find.
    :return: True if the key is found, otherwise False.
    """
    tree.root = tree._splay(tree.root, key)
    return tree.root is not None and tree.root.key == key
def count_splay_steps(tree, key):
    """
    Count the number of steps taken to splay a key to the root of the splay tree.

    :param tree: The SplayTree instance.
    :param key: The key to splay.
    :return: The number of steps taken to splay the key.
    """
    def _splay_with_count(root, key):
        if root is None or root.key == key:
            return root, 0

        steps = 0

        if root.key > key:
            if root.left is None:
                return root, steps
            if root.left.key > key:
                root.left.left, left_steps = _splay_with_count(root.left.left, key)
                root = tree._rotate_right(root)
                steps += left_steps + 1
            elif root.left.key < key:
                root.left.right, right_steps = _splay_with_count(root.left.right, key)
                if root.left.right is not None:
                    root.left = tree._rotate_left(root.left)
                    steps += right_steps + 1
            return (root, steps) if root.left is None else (tree._rotate_right(root), steps + 1)
        else:
            if root.right is None:
                return root, steps
            if root.right.key > key:
                root.right.left, left_steps = _splay_with_count(root.right.left, key)
                if root.right.left is not None:
                    root.right = tree._rotate_right(root.right)
                    steps += left_steps + 1
            elif root.right.key < key:
                root.right.right, right_steps = _splay_with_count(root.right.right, key)
                root = tree._rotate_left(root)
                steps += right_steps + 1
            return (root, steps) if root.right is None else (tree._rotate_left(root), steps + 1)

    tree.root, steps = _splay_with_count(tree.root, key)
    return steps

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Assuming the SplayTree and count_splay_steps are already defined

def simulate_splay_tree_operations(tree, elements, c, m):
    steps_for_x = []
    x = elements[0]  # Element to focus on

    focus_queries = int(c * m)
    other_queries = m - focus_queries

    queries = [x] * focus_queries + [random.choice(elements) for _ in range(other_queries)]
    random.shuffle(queries)

    for query in queries:
        steps = count_splay_steps(tree, query)
        if query == x:
            steps_for_x.append(steps)

    return np.mean(steps_for_x) if steps_for_x else 0

def calculate_average_steps(c_val):
    steps = []
    for n in n_values:
        tree = SplayTree()
        elements_to_insert = list(range(1, n + 1))
        random.shuffle(elements_to_insert)
        for element in elements_to_insert:
            tree.insert(element)
        
        avg_steps = simulate_splay_tree_operations(tree, elements_to_insert, c_val, m)
        steps.append(avg_steps)
    return steps

def update(val):
    c_new = slider.val
    new_steps = calculate_average_steps(c_new)
    global line
    # Add a shadow effect for the previous line
    line.set_alpha(0.1)  # Set the previous line to be more transparent
    line.set_color('#1f77b4')  # Set the color to a lighter shade of blue
    
    # Create a new line for the current c value
    new_line, = ax1.plot(n_values, new_steps, '-', color='#1f77b4', linewidth=2)
    
    # Update the line reference to the new line
    line = new_line
    
    ax1.set_title(f'Splay Tree Average Steps for c = {c_new:.2f}')
    fig.canvas.draw_idle()
# Experiment setup
n_values = range(1, 1001, 50)
m = 5000
initial_c = 0.01

# Initial calculation
initial_steps = calculate_average_steps(initial_c)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

line, = ax1.plot(n_values, initial_steps, '-', color='#1f77b4', linewidth=2)
ax1.set_xlabel('n (Number of Elements)')
ax1.set_ylabel('Average Steps')
ax1.set_title(f'Splay Tree Average Steps for c = {initial_c:.2f}')
ax1.grid(True, linestyle='--', alpha=0.3)

# Slider axis
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(
    ax=ax_slider,
    label='C',
    valmin=0,
    valmax=1.0,
    valinit=initial_c,
    valstep=0.05,
    color='#1f77b4'
)

slider.on_changed(update)

plt.show()
