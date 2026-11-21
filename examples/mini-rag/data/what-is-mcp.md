# What is the Model Context Protocol?

The Model Context Protocol (MCP) is an open standard for connecting AI applications to external tools and data sources. Instead of writing a one-off integration for every model host, you expose capabilities through an MCP server that a compatible client can call.

A small MCP server might wrap a database query, an issue tracker, or a file search. The model never receives your credentials directly; your server runs the tool with the privileges you choose, then returns a result the model can read.

MCP belongs later in a learning path, after you already understand tool calling: the model proposes a function and arguments, your code runs that function, and the result is fed back into the next model call. MCP standardises how those tools are described and reached.
