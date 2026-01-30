# Sample API Inputs for Documentation Generator

This directory contains sample input files for testing the API Documentation Generator.

## File: sample_api_input.json

This file contains a complete example for the **User Registration API**.

### How to Use

1. **Open the file**: `sample_api_input.json`
2. **Copy the values** and paste them into the app.py Streamlit interface:
   - API Name: `User Registration`
   - Method: `POST`
   - Endpoint: `/api/v1/users/register`
   - Description: `Creates a new user account in the system with email verification`
   - Parameters: Copy the entire JSON parameters object

3. **Click "Generate Documentation"** in the Streamlit app

## Additional Examples

### Example 1: Get User Profile
```json
{
  "api_name": "Get User Profile",
  "method": "GET",
  "endpoint": "/api/v1/users/{userId}",
  "description": "Retrieves detailed information about a specific user by their ID",
  "parameters": {
    "userId": "string - Unique identifier for the user (path parameter, required)",
    "include_private": "boolean - Whether to include private information (query parameter, optional, default: false)"
  }
}
```

### Example 2: Update Order Status
```json
{
  "api_name": "Update Order Status",
  "method": "PUT",
  "endpoint": "/api/v1/orders/{orderId}/status",
  "description": "Updates the status of an existing order",
  "parameters": {
    "orderId": "string - Order ID (path parameter, required)",
    "status": "string - New order status: pending, processing, shipped, delivered, cancelled (required)",
    "tracking_number": "string - Shipping tracking number (optional, required if status is 'shipped')",
    "notes": "string - Additional notes about the status change (optional)",
    "notify_customer": "boolean - Send notification to customer (optional, default: true)"
  }
}
```

### Example 3: Search Products
```json
{
  "api_name": "Search Products",
  "method": "GET",
  "endpoint": "/api/v1/products/search",
  "description": "Search for products using various filters and return paginated results",
  "parameters": {
    "query": "string - Search query string (optional)",
    "category": "string - Product category filter (optional)",
    "min_price": "number - Minimum price filter (optional)",
    "max_price": "number - Maximum price filter (optional)",
    "in_stock": "boolean - Filter for in-stock items only (optional, default: false)",
    "sort_by": "string - Sort field: name, price, created_date (optional, default: name)",
    "order": "string - Sort order: asc or desc (optional, default: asc)",
    "page": "integer - Page number for pagination (optional, default: 1)",
    "limit": "integer - Items per page (optional, default: 20, max: 100)"
  }
}
```

### Example 4: Delete Resource
```json
{
  "api_name": "Delete User Account",
  "method": "DELETE",
  "endpoint": "/api/v1/users/{userId}",
  "description": "Permanently deletes a user account and all associated data",
  "parameters": {
    "userId": "string - Unique identifier for the user to delete (path parameter, required)",
    "confirmation_token": "string - Security token to confirm deletion (required)",
    "transfer_data_to": "string - User ID to transfer data ownership to (optional)",
    "send_confirmation_email": "boolean - Send deletion confirmation email (optional, default: true)"
  }
}
```

### Example 5: File Upload
```json
{
  "api_name": "Upload Document",
  "method": "POST",
  "endpoint": "/api/v1/documents/upload",
  "description": "Uploads a document file to the system with metadata",
  "parameters": {
    "file": "file - The document file to upload (required, max 10MB, formats: pdf, doc, docx, txt)",
    "title": "string - Document title (required)",
    "description": "string - Document description (optional)",
    "category": "string - Document category (required)",
    "tags": "array - Array of tag strings (optional)",
    "is_public": "boolean - Whether document is publicly accessible (optional, default: false)",
    "expiry_date": "string - Document expiration date in ISO format (optional)"
  }
}
```

## Testing Tips

1. **Start Simple**: Begin with the main sample file (sample_api_input.json)
2. **Test Different Methods**: Try GET, POST, PUT, DELETE to see different documentation styles
3. **Vary Complexity**: Test with simple APIs (1-2 parameters) and complex ones (10+ parameters)
4. **Save Results**: Use the "Save/Load" tab in the app to persist your generated documentation
5. **Compare Outputs**: Generate docs for similar APIs and compare the AI's approach

## Notes

- All examples use JSON format for parameters
- The AI will generate comprehensive documentation including request/response examples, error codes, and usage notes
- You can modify any example to fit your specific API requirements
