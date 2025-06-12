# Effectively Using Prompts and Tools with the MCP Protocol

The Model Context Protocol (MCP) is a powerful framework for building AI-driven applications that need to orchestrate prompts, tools, and resources in a structured, extensible way. In this post, we'll explore how to leverage MCP's prompt and tool features to create robust, context-aware LLM workflows.

---

## What Are Prompts and Tools in MCP?

- **Prompts** are reusable message templates or instructions that can be parameterized and injected into the LLM's context.
- **Tools** are callable functions or APIs that the LLM can invoke to perform actions, fetch data, or process information.

MCP treats these as first-class citizens, but leaves it up to you to orchestrate how and when they're used together.

---

## Why Combine Prompts and Tools?

Combining prompts and tools allows you to:

- Provide the LLM with clear, structured context before it uses a tool.
- Standardize tool usage instructions, improving reliability and user experience.
- Enable dynamic, context-aware tool invocation in your AI applications.

---

## Pattern: Binding Prompts to Tools

The key pattern is **automatic prompt injection before tool execution**. Using a naming convention (e.g., `tool:<toolName>`), the client automatically fetches and injects the associated prompt whenever the LLM decides to use a tool.

**The Flow:**

1. 🤖 LLM analyzes query and decides to use a tool
2. 💬 Client automatically injects the associated prompt (provides context)
3. 🔧 Tool executes with proper context from the prompt
4. ✅ Results returned with optimal context

This ensures every tool call has the contextual knowledge it needs to perform optimally.

### **Server-Side: Registering Tools and Prompts**

```typescript
import { McpServer } from "your-mcp-sdk/server/mcp.js";
import { z } from "zod";

// Register a tool
server.tool(
  "summarize",
  "Summarizes a given text",
  { text: z.string() },
  async ({ text }) => ({
    content: [{ type: "text", text: `Summary: ...` }],
  })
);

// Register the associated prompt
server.prompt(
  "tool:summarize",
  "Prompt for the summarize tool",
  { text: z.string() },
  async ({ text }) => ({
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text: `Please summarize the following text:\n\n${text}`,
        },
      },
    ],
  })
);
```

---

### **Client-Side: Simple Implementation Focusing on Prompt and Tool Binding**

Here's a simplified client that demonstrates the core concepts of binding prompts and tools in MCP:

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { OpenAI } from "openai";

interface OpenAIMessage {
  role: "system" | "user" | "assistant";
  content: string | any;
}

interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: any;
  };
}

/**
 * Simple MCP Client demonstrating prompt and tool binding
 */
class SimpleMCPClient {
  private client: Client;
  private openai: OpenAI;
  private availableTools: OpenAITool[] = [];
  private conversation: OpenAIMessage[] = [];

  constructor() {
    this.client = new Client({
      name: "simple-mcp-client",
      version: "1.0.0",
    });

    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  /**
   * Connect to MCP server and load tools
   */
  async connect(command: string, args: string[] = []) {
    const transport = new StdioClientTransport({ command, args });
    await this.client.connect(transport);

    // Load and bind tools to LLM
    await this.loadTools();
    console.log(`✅ Connected! Found ${this.availableTools.length} tools`);
  }

  /**
   * Load tools from MCP server and convert to OpenAI format
   */
  private async loadTools() {
    const response = await this.client.listTools();

    this.availableTools = response.tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description || tool.name,
        parameters: tool.inputSchema || { type: "object", properties: {} },
      },
    }));
  }

  /**
   * Process query with automatic tool binding and execution
   */
  async processQuery(query: string): Promise<string> {
    // Add user message
    this.conversation.push({ role: "user", content: query });

    // Call LLM with tools bound
    const response = await this.openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: this.conversation,
      tools: this.availableTools,
      tool_choice: "auto",
    });

    const message = response.choices[0].message;

    // Handle tool calls if present
    if (message.tool_calls?.length) {
      // Add assistant message with tool calls
      this.conversation.push({
        role: "assistant",
        content: message.content,
        tool_calls: message.tool_calls,
      });

      // Execute each tool with automatic prompt injection
      for (const toolCall of message.tool_calls) {
        // 1. FIRST: Automatically inject the associated prompt
        await this.injectToolPrompt(
          toolCall.function.name,
          JSON.parse(toolCall.function.arguments)
        );

        // 2. THEN: Execute the tool with prompt context
        const result = await this.client.callTool({
          name: toolCall.function.name,
          arguments: JSON.parse(toolCall.function.arguments),
        });

        // 3. Add tool result to conversation
        this.conversation.push({
          role: "user",
          content: JSON.stringify(result.content),
          tool_call_id: toolCall.id,
        });
      }

      // Get final response after tool execution
      const finalResponse = await this.openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: this.conversation,
        tools: this.availableTools,
        tool_choice: "auto",
      });

      const finalMessage = finalResponse.choices[0].message.content;
      this.conversation.push({ role: "assistant", content: finalMessage });
      return finalMessage;
    } else {
      // No tools used - just add response
      this.conversation.push({ role: "assistant", content: message.content });
      return message.content;
    }
  }

  /**
   * Automatically inject the associated prompt before tool execution
   * This is the key method that implements the prompt-tool binding logic
   */
  private async injectToolPrompt(toolName: string, toolArgs: any) {
    try {
      // Try to get the associated prompt using naming convention
      const promptResult = await this.client.getPrompt({
        name: `tool:${toolName}`,
        arguments: toolArgs,
      });

      console.log(`💬 Auto-injecting prompt for tool: ${toolName}`);

      // Add prompt messages to conversation BEFORE tool execution
      for (const msg of promptResult.messages) {
        this.conversation.push({
          role: msg.role,
          content:
            typeof msg.content === "string" ? msg.content : msg.content.text,
        });
      }
    } catch (error) {
      // Prompt not found or not available - proceed without it
      console.log(
        `ℹ️  No prompt found for tool: ${toolName}, proceeding without prompt`
      );
    }
  }

  async disconnect() {
    await this.client.close();
  }
}

/**
 * Example usage demonstrating prompt and tool binding
 */
async function main() {
  const client = new SimpleMCPClient();

  try {
    // Connect to filesystem server
    await client.connect("npx", [
      "-y",
      "@modelcontextprotocol/server-filesystem",
      "/tmp",
    ]);

    // Example 1: Query that will use tools automatically
    console.log("\n📁 Example 1: Direct tool usage");
    const response1 = await client.processQuery(
      "List the files in the current directory"
    );
    console.log("Response:", response1);

    // Example 2: Another query that will automatically use prompts with tools
    console.log("\n💬 Example 2: Query with automatic prompt injection");
    const response2 = await client.processQuery(
      "Analyze the directory structure and tell me what you find"
    );
    console.log("Response:", response2);

    // Example 3: Complex query requiring multiple tools
    console.log("\n🔧 Example 3: Complex query with multiple tools");
    const response3 = await client.processQuery(
      "Find all .txt files and read the contents of the first one you find"
    );
    console.log("Response:", response3);
  } catch (error) {
    console.error("Error:", error);
  } finally {
    await client.disconnect();
  }
}

// Run the example
if (require.main === module) {
  main();
}

export { SimpleMCPClient };
```

**Key Concepts Demonstrated:**

### **1. Tool Binding (Core Concept)**

```typescript
// Load tools from MCP server
const response = await this.client.listTools();

// Convert to OpenAI format
this.availableTools = response.tools.map((tool) => ({
  type: "function",
  function: {
    name: tool.name,
    description: tool.description,
    parameters: tool.inputSchema,
  },
}));

// Bind to LLM in every call
const response = await this.openai.chat.completions.create({
  tools: this.availableTools, // ← Tools are bound here
  tool_choice: "auto",
});
```

### **2. Prompt Integration**

```typescript
// Get prompt from MCP server
const promptResult = await this.client.getPrompt({
  name: promptName,
  arguments: {},
});

// Add to conversation before processing
for (const msg of promptResult.messages) {
  this.conversation.push({ role: msg.role, content: msg.content });
}
```

### **3. Automatic Tool Execution**

```typescript
// LLM requests tools → Execute automatically
if (message.tool_calls?.length) {
  for (const toolCall of message.tool_calls) {
    const result = await this.client.callTool({
      name: toolCall.function.name,
      arguments: JSON.parse(toolCall.function.arguments),
    });
  }
}
```

**Usage:**

```bash
# Install dependencies
npm install openai @modelcontextprotocol/sdk

# Set API key
export OPENAI_API_KEY="your-key"

# Run
node client.js
```

**What this demonstrates:**

- ✅ **Simple tool binding**: Tools automatically available to LLM
- ✅ **Prompt integration**: How to inject prompts before tool usage
- ✅ **Automatic execution**: LLM decides when and how to use tools
- ✅ **Clean code**: Focused on core concepts without complexity

This simplified version shows the essential pattern for binding prompts and tools in MCP!

---

## Key Takeaways

- **MCP provides the building blocks** for prompts and tools, but orchestration is up to your application logic.
- **Use naming conventions** to bind prompts to tools for discoverability and consistency.
- **Always inject the prompt into the LLM context** before allowing a tool call—this ensures the LLM has the right instructions and context.
- **This pattern improves reliability, user experience, and maintainability** of your AI-powered applications.

---

## Advanced Tips

- **Dynamic Prompts:** You can update, enable, disable, or remove prompts at runtime using MCP's prompt management features.
- **Notifications:** Use MCP's prompt list change notifications to keep clients in sync with available prompts.
- **Extensibility:** This pattern works for any number of tools and can be extended to resources or other MCP features.

---

## Conclusion

By thoughtfully combining prompts and tools in MCP, you can build smarter, more reliable, and more maintainable LLM applications. The key is to orchestrate the flow so that the LLM always receives the right context before taking action—unlocking the full power of both prompts and tools.

---

**Ready to try it?**  
Check out the [MCP documentation](#) or dive into your own project using these patterns!
