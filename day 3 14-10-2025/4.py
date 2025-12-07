class Queue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []
    
    def isEmpty(self):
        return len(self.queue) == 0
    
    def isFull(self):
        return len(self.queue) == self.capacity
    
    def enqueue(self, value):
        if self.isFull():
            print("The queue is full already!")
            return None
        else:
            self.queue.append(value)

    def dequeue(self):
        if self.isEmpty():
            print("The queue is empty already!")
            return None
        else:
            return self.queue.pop(0)
    
    def front(self):
        if self.isEmpty():
            print("The queue is empty already!")
            return None
        else:
            return self.queue[0]

    def show(self):
        return self.queue
    
    def size(self):
        return len(self.queue)
    
my_queue = Queue(4)
print("Is queue empty?", my_queue.isEmpty())
print("Is queue full?", my_queue.isFull())
my_queue.enqueue(1)
my_queue.enqueue(2)
my_queue.enqueue(3)
print("Queue: ", my_queue.show())
print("Is queue full?", my_queue.isFull())
my_queue.dequeue()
print("Front element:", my_queue.front())
print("All elements:", my_queue.show())
print("Queue size:", my_queue.size())