# Waht we do
We will now develop a program using Python and lambda.


# **Project Development Guide**

Welcome, developer\! This document outlines the architectural principles and development workflow for this project. Adhering to these guidelines is crucial for maintaining the system's quality, scalability, and maintainability.

Our philosophy is simple: we build stable, adaptable, and long-lasting software by placing business logic at the core and ensuring all dependencies point inward.

## **Core Principles**

Our architecture is built upon four fundamental principles.

### **1\. Single Responsibility Modules**

The smallest unit of our system is a **module** (e.g., a class, a function set) that is responsible for a single, well-defined piece of functionality. This adheres to the **Single Responsibility Principle (SRP)**.

**Why?** This makes our code easier to understand, test, and refactor without causing unintended side effects. Each piece does one thing and does it well.

### **2\. Test-Driven Development (TDD)**

We do not write implementation code until we have a failing test that defines our goal. Development of any module begins by defining its required functionality as a **"red" test**.

**Why?** This practice forces us to clearly define a module's requirements before implementation. The test suite becomes an executable specification, guaranteeing quality and preventing regressions.

### **3\. Dependency Injection & Loose Coupling**

The system is composed of independent modules wired together at runtime. This is achieved through **Dependency Injection (DI)** via **Interfaces (Abstractions)**. Modules do not create their dependencies; they are given to them.

**Why?** This inverts the flow of control and decouples our modules. It allows us to easily swap implementations (e.g., switching a real database for an in-memory mock during tests) and promotes a modular, plug-and-play architecture.

### **4\. Business Logic is King**

The most critical principle is that **all dependencies must point towards the core business logic**. The business logic, which represents the system's true value, should have no knowledge of external concerns like databases, frameworks, or UI.

**Why?** This protects our core logic from technological churn. We can change a web framework, switch a database, or add a new interface without touching the stable, well-tested core. This is the essence of **Clean Architecture**.

## **Architecture Overview**

We follow a layered architecture (often called Clean Architecture or Onion Architecture) to enforce these principles.

\+-----------------------------------------------------------------+  
|  FRAMEWORKS & DRIVERS (Web, UI, DB Implementations, etc.)       |  
|   \+---------------------------------------------------------+   |  
|   |  INTERFACE ADAPTERS (Controllers, Presenters, Repos)    |   |  
|   |   \+-------------------------------------------------+   |   |  
|   |   |  APPLICATION BUSINESS RULES (Use Cases)         |   |   |  
|   |   |   \+-----------------------------------------+   |   |   |  
|   |   |   |      ENTERPRISE BUSINESS RULES          |   |   |   |  
|   |   |   |      (Entities & Core Domain Logic)     |   |   |   |  
|   |   |   \+-----------------------------------------+   |   |   |  
|   |   |                                                 |   |   |  
|   |   \+-------------------------------------------------+   |   |  
|   |                                                         |   |  
|   \+---------------------------------------------------------+   |  
|                                                                 |  
\+-----------------------------------------------------------------+

\---------------------\> DEPENDENCY RULE \---------------------\>

* **Enterprise Business Rules (Entities):** The absolute core. Contains domain objects and their logic. Has zero dependencies.  
* **Application Business Rules (Use Cases):** Orchestrates the entities to perform application-specific tasks. Defines interfaces that the outer layers must implement (e.g., IUserRepository).  
* **Interface Adapters:** Converts data from the format most convenient for the use cases to the format most convenient for external agencies like the Database or the Web. This is where Repositories (implementing the interfaces from the Use Case layer) and Controllers live.  
* **Frameworks & Drivers:** The outermost layer. Contains the specific tools and technologies (e.g., Express.js, React, PostgreSQL drivers). This layer is volatile and expected to change.

**The Golden Rule:** Code in an inner circle can **NEVER** reference anything in an outer circle.

## **Development Workflow: Adding a New Feature**

Follow these steps to develop in alignment with our architecture.

Let's imagine we are adding a feature: "Register a new User".

### **Step 1: Define the Use Case and Interfaces (Application Layer)**

First, think about the application's goal. What does it need to do? It needs to save a user. How it's saved (SQL, NoSQL) is an implementation detail we ignore for now.

Define the interface for the dependency you'll need.

// src/application/repositories/IUserRepository.ts  
export interface IUserRepository {  
  save(user: User): Promise\<void\>;  
  findByEmail(email: string): Promise\<User | null\>;  
}

### **Step 2: Write a Failing Test for the Use Case (Red)**

Now, write a test for the use case. This test will use a mock (or "fake") implementation of the repository interface.

// src/application/usecases/RegisterUser.test.ts  
import { RegisterUserUseCase } from './RegisterUser';  
import { InMemoryUserRepository } from '../../tests/mocks/InMemoryUserRepository';

test('should register a new user', async () \=\> {  
  const userRepository \= new InMemoryUserRepository();  
  const registerUser \= new RegisterUserUseCase(userRepository); // DI in action\!

  const input \= { name: 'John Doe', email: 'john@example.com' };  
  await registerUser.execute(input);

  const savedUser \= await userRepository.findByEmail('john@example.com');  
  expect(savedUser).not.toBeNull();  
  expect(savedUser?.name).toBe('John Doe');  
});

*This test will fail because RegisterUserUseCase doesn't exist yet.*

### **Step 3: Write the Implementation to Pass the Test (Green)**

Create the use case class and the simplest code required to make the test pass.

// src/application/usecases/RegisterUser.ts  
import { User } from '../../domain/User';  
import { IUserRepository } from '../repositories/IUserRepository';

export class RegisterUserUseCase {  
  constructor(private userRepository: IUserRepository) {} // Dependency is injected

  async execute(input: { name: string; email: string }): Promise\<void\> {  
    const existingUser \= await this.userRepository.findByEmail(input.email);  
    if (existingUser) {  
      throw new Error('Email already in use.');  
    }  
    const user \= new User(input.name, input.email);  
    await this.userRepository.save(user);  
  }  
}

Run the test again. It should now pass.

### **Step 4: Refactor**

Look at your use case and test code. Is it clean? Is it readable? Can anything be improved? Refactor with confidence, knowing your tests will catch any regressions.

### **Step 5: Implement the Outer Layers (Infrastructure)**

Now, and only now, do you think about technology.

1. Create the Repository Implementation (Interface Adapter Layer):  
   This class will implement the IUserRepository interface and contain the actual database logic.  
   // src/infrastructure/db/PostgresUserRepository.ts  
   import { IUserRepository } from '../../application/repositories/IUserRepository';  
   // ... imports for db client and User entity

   export class PostgresUserRepository implements IUserRepository {  
     async save(user: User): Promise\<void\> {  
       // ... actual postgres client code to save the user  
     }  
     // ... other methods  
   }

2. Create the Controller (Interface Adapter Layer):  
   This will handle the HTTP request and call the use case.  
   // src/infrastructure/web/UserController.ts  
   export class UserController {  
     constructor(private registerUserUseCase: RegisterUserUseCase) {}

     async handleRegister(req: Request, res: Response) {  
       try {  
         await this.registerUserUseCase.execute(req.body);  
         res.status(201).send();  
       } catch (error) {  
         res.status(400).json({ message: error.message });  
       }  
     }  
   }

3. Wire Everything Together (Main/App Entry Point):  
   In your application's main entry point, you will perform the Dependency Injection.  
   // src/main.ts  
   const userRepo \= new PostgresUserRepository();  
   const registerUseCase \= new RegisterUserUseCase(userRepo);  
   const userController \= new UserController(registerUseCase);

   // app.post('/users', userController.handleRegister);

## **Key Conventions**

* **Interfaces:** Name interfaces with an I prefix (e.g., IUserRepository).  
* **File Location:** Place files in the appropriate architectural layer folder (domain, application, infrastructure).  
* **Testing:** Unit tests should reside alongside the code they test or in a dedicated \_\_tests\_\_ folder. Mocks and stubs belong in a test-specific directory.