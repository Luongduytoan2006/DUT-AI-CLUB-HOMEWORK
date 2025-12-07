class Stack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.stack = []
        

    def isEmpty(self):
        return len(self.stack) == 0
    
    def isFull(self):
        return len(self.stack) == self.capacity
    
    def push(self, value):
        if self.isFull():
            print("The stack is full already!")
            return None
        else:
            self.stack.append(value)

    def pop(self):
        if self.isEmpty():
            print("The stack is empty already!")
            return None
        else:
            return self.stack.pop()
            
    def top(self):
        if self.isEmpty():
            print("The stack is empty already!")
            return None
        else:
            return self.stack[-1]
        
    def show(self):
        return self.stack
    
    def size(self):
        return len(self.stack)
        

my_stack = Stack(4)
print("Is stack empty?", my_stack.isEmpty())
print("Is stack full?", my_stack.isFull())
my_stack.push(1)
my_stack.push(2)
print("Is stack empty after pushing 2 elements?", my_stack.isEmpty())
print("Is stack full after pushing 2 elements?", my_stack.isFull())
my_stack.push(3)
my_stack.push(4)
print("Is stack full after pushing 4 elements?", my_stack.isFull())
print("Top element:", my_stack.top())
my_stack.push(5)
print("Popped element:", my_stack.pop())
print("Top element:", my_stack.top())
print("All elements:", my_stack.show())
print("Stack size:", my_stack.size())
