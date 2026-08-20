# MindSpace V1 Product Plan

## Positioning

MindSpace turns a folder of learning or work materials into a searchable, traceable AI knowledge space.

## V1 Screens

1. Home
   - Recent knowledge spaces
   - Create a new space
2. Space Detail
   - Space summary
   - Search box
   - Search results with source location
3. Upload
   - File import entry
   - Supported formats notice

## V1 Backend

1. `GET /spaces`
2. `POST /spaces`
3. `POST /spaces/{space_id}/files`
4. `POST /spaces/{space_id}/search`

## UI Direction

- Young and clean
- iOS-like spacing, rounded panels, light surface
- Product prototype ready for demo
- Calm blue as the product action color
- Search-first interaction

## Next Iterations

1. Connect real mobile document picker.
2. Add semantic search with embeddings.
3. Add RAG answer generation with citations.
4. Add audio/video transcription search.
5. Add Figma screen export after a Figma file is provided.
