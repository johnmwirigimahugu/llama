<?php
/**
 * Llama.php - A single-file PHP AI framework inspired by LLPhant
 * Supports OpenAI, Anthropic, Ollama, and local vector storage
 * 
 * Usage Examples:
 * 
 * // Simple chat
 * $chat = new LlamaChat('openai');
 * echo $chat->ask("What is PHP?");
 * 
 * // RAG (Question Answering)
 * $rag = new LlamaRAG();
 * $rag->addDocument("PHP is a programming language", "php-info");
 * echo $rag->ask("What is PHP?");
 * 
 * // Embeddings
 * $embedder = new LlamaEmbeddings('openai');
 * $vector = $embedder->embed("Hello world");
 */

// =============================================================================
// CORE INTERFACES
// =============================================================================

interface LlamaChatInterface {
    public function ask(string $question): string;
    public function setSystemMessage(string $message): void;
}

interface LlamaEmbeddingInterface {
    public function embed(string $text): array;
    public function embedBatch(array $texts): array;
}

interface LlamaVectorStoreInterface {
    public function add(string $id, array $vector, string $content, array $metadata = []): void;
    public function search(array $vector, int $limit = 5): array;
    public function get(string $id): ?array;
}

// =============================================================================
// CONFIGURATION CLASSES
// =============================================================================

class LlamaConfig {
    public static array $settings = [
        'openai_api_key' => null,
        'anthropic_api_key' => null,
        'ollama_url' => 'http://localhost:11434',
        'default_model' => [
            'openai' => 'gpt-3.5-turbo',
            'anthropic' => 'claude-3-haiku-20240307',
            'ollama' => 'llama2'
        ],
        'embedding_model' => [
            'openai' => 'text-embedding-ada-002',
            'ollama' => 'nomic-embed-text'
        ]
    ];
    
    public static function set(string $key, $value): void {
        self::$settings[$key] = $value;
    }
    
    public static function get(string $key, $default = null) {
        return self::$settings[$key] ?? $default;
    }
    
    public static function loadFromEnv(): void {
        self::$settings['openai_api_key'] = $_ENV['OPENAI_API_KEY'] ?? getenv('OPENAI_API_KEY') ?: null;
        self::$settings['anthropic_api_key'] = $_ENV['ANTHROPIC_API_KEY'] ?? getenv('ANTHROPIC_API_KEY') ?: null;
        self::$settings['ollama_url'] = $_ENV['OLLAMA_URL'] ?? getenv('OLLAMA_URL') ?: 'http://localhost:11434';
    }
}

// =============================================================================
// HTTP CLIENT
// =============================================================================

class LlamaHttpClient {
    private int $timeout;
    
    public function __construct(int $timeout = 30) {
        $this->timeout = $timeout;
    }
    
    public function post(string $url, array $data, array $headers = []): array {
        $ch = curl_init();
        curl_setopt_array($ch, [
            CURLOPT_URL => $url,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($data),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $this->timeout,
            CURLOPT_HTTPHEADER => array_merge([
                'Content-Type: application/json'
            ], $headers)
        ]);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        if ($error) {
            throw new Exception("HTTP Error: $error");
        }
        
        if ($httpCode >= 400) {
            throw new Exception("HTTP Error $httpCode: $response");
        }
        
        $decoded = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new Exception("JSON decode error: " . json_last_error_msg());
        }
        
        return $decoded;
    }
    
    public function get(string $url, array $headers = []): array {
        $ch = curl_init();
        curl_setopt_array($ch, [
            CURLOPT_URL => $url,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $this->timeout,
            CURLOPT_HTTPHEADER => $headers
        ]);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode >= 400) {
            throw new Exception("HTTP Error $httpCode: $response");
        }
        
        return json_decode($response, true) ?? [];
    }
}

// =============================================================================
// DOCUMENT HANDLING
// =============================================================================

class LlamaDocument {
    public string $id;
    public string $content;
    public array $metadata;
    public ?array $embedding;
    
    public function __construct(string $content, string $id = null, array $metadata = []) {
        $this->id = $id ?? uniqid('doc_');
        $this->content = $content;
        $this->metadata = $metadata;
        $this->embedding = null;
    }
}

class LlamaDocumentSplitter {
    public static function split(string $text, int $chunkSize = 1000, int $overlap = 200): array {
        $chunks = [];
        $words = explode(' ', $text);
        $totalWords = count($words);
        
        for ($i = 0; $i < $totalWords; $i += ($chunkSize - $overlap)) {
            $chunk = array_slice($words, $i, $chunkSize);
            $chunkText = implode(' ', $chunk);
            
            if (!empty(trim($chunkText))) {
                $chunks[] = new LlamaDocument($chunkText, uniqid('chunk_'));
            }
            
            if ($i + $chunkSize >= $totalWords) {
                break;
            }
        }
        
        return $chunks;
    }
}

// =============================================================================
// CHAT IMPLEMENTATIONS
// =============================================================================

class LlamaOpenAIChat implements LlamaChatInterface {
    private LlamaHttpClient $client;
    private string $model;
    private ?string $systemMessage = null;
    private array $messages = [];
    
    public function __construct(string $model = null) {
        $this->client = new LlamaHttpClient();
        $this->model = $model ?? LlamaConfig::get('default_model')['openai'];
    }
    
    public function setSystemMessage(string $message): void {
        $this->systemMessage = $message;
    }
    
    public function ask(string $question): string {
        $messages = [];
        
        if ($this->systemMessage) {
            $messages[] = ['role' => 'system', 'content' => $this->systemMessage];
        }
        
        // Add conversation history
        $messages = array_merge($messages, $this->messages);
        $messages[] = ['role' => 'user', 'content' => $question];
        
        $response = $this->client->post('https://api.openai.com/v1/chat/completions', [
            'model' => $this->model,
            'messages' => $messages,
            'temperature' => 0.7
        ], [
            'Authorization: Bearer ' . LlamaConfig::get('openai_api_key')
        ]);
        
        $answer = $response['choices'][0]['message']['content'] ?? '';
        
        // Store in conversation history
        $this->messages[] = ['role' => 'user', 'content' => $question];
        $this->messages[] = ['role' => 'assistant', 'content' => $answer];
        
        return $answer;
    }
}

class LlamaAnthropicChat implements LlamaChatInterface {
    private LlamaHttpClient $client;
    private string $model;
    private ?string $systemMessage = null;
    private array $messages = [];
    
    public function __construct(string $model = null) {
        $this->client = new LlamaHttpClient();
        $this->model = $model ?? LlamaConfig::get('default_model')['anthropic'];
    }
    
    public function setSystemMessage(string $message): void {
        $this->systemMessage = $message;
    }
    
    public function ask(string $question): string {
        $messages = array_merge($this->messages, [
            ['role' => 'user', 'content' => $question]
        ]);
        
        $payload = [
            'model' => $this->model,
            'max_tokens' => 1000,
            'messages' => $messages
        ];
        
        if ($this->systemMessage) {
            $payload['system'] = $this->systemMessage;
        }
        
        $response = $this->client->post('https://api.anthropic.com/v1/messages', $payload, [
            'x-api-key: ' . LlamaConfig::get('anthropic_api_key'),
            'anthropic-version: 2023-06-01'
        ]);
        
        $answer = $response['content'][0]['text'] ?? '';
        
        // Store in conversation history
        $this->messages[] = ['role' => 'user', 'content' => $question];
        $this->messages[] = ['role' => 'assistant', 'content' => $answer];
        
        return $answer;
    }
}

class LlamaOllamaChat implements LlamaChatInterface {
    private LlamaHttpClient $client;
    private string $model;
    private ?string $systemMessage = null;
    private array $messages = [];
    
    public function __construct(string $model = null) {
        $this->client = new LlamaHttpClient();
        $this->model = $model ?? LlamaConfig::get('default_model')['ollama'];
    }
    
    public function setSystemMessage(string $message): void {
        $this->systemMessage = $message;
    }
    
    public function ask(string $question): string {
        $messages = [];
        
        if ($this->systemMessage) {
            $messages[] = ['role' => 'system', 'content' => $this->systemMessage];
        }
        
        $messages = array_merge($messages, $this->messages);
        $messages[] = ['role' => 'user', 'content' => $question];
        
        $response = $this->client->post(LlamaConfig::get('ollama_url') . '/api/chat', [
            'model' => $this->model,
            'messages' => $messages,
            'stream' => false
        ]);
        
        $answer = $response['message']['content'] ?? '';
        
        // Store in conversation history
        $this->messages[] = ['role' => 'user', 'content' => $question];
        $this->messages[] = ['role' => 'assistant', 'content' => $answer];
        
        return $answer;
    }
}

// =============================================================================
// EMBEDDING IMPLEMENTATIONS
// =============================================================================

class LlamaOpenAIEmbeddings implements LlamaEmbeddingInterface {
    private LlamaHttpClient $client;
    private string $model;
    
    public function __construct(string $model = null) {
        $this->client = new LlamaHttpClient();
        $this->model = $model ?? LlamaConfig::get('embedding_model')['openai'];
    }
    
    public function embed(string $text): array {
        $response = $this->client->post('https://api.openai.com/v1/embeddings', [
            'model' => $this->model,
            'input' => $text
        ], [
            'Authorization: Bearer ' . LlamaConfig::get('openai_api_key')
        ]);
        
        return $response['data'][0]['embedding'] ?? [];
    }
    
    public function embedBatch(array $texts): array {
        $response = $this->client->post('https://api.openai.com/v1/embeddings', [
            'model' => $this->model,
            'input' => $texts
        ], [
            'Authorization: Bearer ' . LlamaConfig::get('openai_api_key')
        ]);
        
        return array_map(fn($item) => $item['embedding'], $response['data'] ?? []);
    }
}

class LlamaOllamaEmbeddings implements LlamaEmbeddingInterface {
    private LlamaHttpClient $client;
    private string $model;
    
    public function __construct(string $model = null) {
        $this->client = new LlamaHttpClient();
        $this->model = $model ?? LlamaConfig::get('embedding_model')['ollama'];
    }
    
    public function embed(string $text): array {
        $response = $this->client->post(LlamaConfig::get('ollama_url') . '/api/embeddings', [
            'model' => $this->model,
            'prompt' => $text
        ]);
        
        return $response['embedding'] ?? [];
    }
    
    public function embedBatch(array $texts): array {
        $embeddings = [];
        foreach ($texts as $text) {
            $embeddings[] = $this->embed($text);
        }
        return $embeddings;
    }
}

// =============================================================================
// VECTOR STORE IMPLEMENTATIONS
// =============================================================================

class LlamaMemoryVectorStore implements LlamaVectorStoreInterface {
    private array $vectors = [];
    
    public function add(string $id, array $vector, string $content, array $metadata = []): void {
        $this->vectors[$id] = [
            'id' => $id,
            'vector' => $vector,
            'content' => $content,
            'metadata' => $metadata
        ];
    }
    
    public function search(array $vector, int $limit = 5): array {
        $similarities = [];
        
        foreach ($this->vectors as $item) {
            $similarity = $this->cosineSimilarity($vector, $item['vector']);
            $similarities[] = array_merge($item, ['similarity' => $similarity]);
        }
        
        // Sort by similarity (descending)
        usort($similarities, fn($a, $b) => $b['similarity'] <=> $a['similarity']);
        
        return array_slice($similarities, 0, $limit);
    }
    
    public function get(string $id): ?array {
        return $this->vectors[$id] ?? null;
    }
    
    private function cosineSimilarity(array $a, array $b): float {
        if (count($a) !== count($b)) {
            return 0.0;
        }
        
        $dotProduct = 0.0;
        $normA = 0.0;
        $normB = 0.0;
        
        for ($i = 0; $i < count($a); $i++) {
            $dotProduct += $a[$i] * $b[$i];
            $normA += $a[$i] * $a[$i];
            $normB += $b[$i] * $b[$i];
        }
        
        if ($normA == 0.0 || $normB == 0.0) {
            return 0.0;
        }
        
        return $dotProduct / (sqrt($normA) * sqrt($normB));
    }
    
    public function count(): int {
        return count($this->vectors);
    }
    
    public function clear(): void {
        $this->vectors = [];
    }
}

class LlamaFileVectorStore implements LlamaVectorStoreInterface {
    private string $filename;
    private array $vectors = [];
    private bool $loaded = false;
    
    public function __construct(string $filename = 'vectors.json') {
        $this->filename = $filename;
    }
    
    private function load(): void {
        if ($this->loaded) return;
        
        if (file_exists($this->filename)) {
            $data = json_decode(file_get_contents($this->filename), true);
            $this->vectors = $data ?? [];
        }
        $this->loaded = true;
    }
    
    private function save(): void {
        file_put_contents($this->filename, json_encode($this->vectors, JSON_PRETTY_PRINT));
    }
    
    public function add(string $id, array $vector, string $content, array $metadata = []): void {
        $this->load();
        $this->vectors[$id] = [
            'id' => $id,
            'vector' => $vector,
            'content' => $content,
            'metadata' => $metadata
        ];
        $this->save();
    }
    
    public function search(array $vector, int $limit = 5): array {
        $this->load();
        $memoryStore = new LlamaMemoryVectorStore();
        
        foreach ($this->vectors as $item) {
            $memoryStore->add($item['id'], $item['vector'], $item['content'], $item['metadata']);
        }
        
        return $memoryStore->search($vector, $limit);
    }
    
    public function get(string $id): ?array {
        $this->load();
        return $this->vectors[$id] ?? null;
    }
}

// =============================================================================
// MAIN CLASSES
// =============================================================================

class LlamaChat {
    private LlamaChatInterface $chat;
    
    public function __construct(string $provider = 'openai', string $model = null) {
        LlamaConfig::loadFromEnv();
        
        switch (strtolower($provider)) {
            case 'openai':
                if (!LlamaConfig::get('openai_api_key')) {
                    throw new Exception('OpenAI API key not set. Set OPENAI_API_KEY environment variable or use LlamaConfig::set("openai_api_key", "your-key")');
                }
                $this->chat = new LlamaOpenAIChat($model);
                break;
            case 'anthropic':
                if (!LlamaConfig::get('anthropic_api_key')) {
                    throw new Exception('Anthropic API key not set. Set ANTHROPIC_API_KEY environment variable or use LlamaConfig::set("anthropic_api_key", "your-key")');
                }
                $this->chat = new LlamaAnthropicChat($model);
                break;
            case 'ollama':
                $this->chat = new LlamaOllamaChat($model);
                break;
            default:
                throw new Exception("Unsupported provider: $provider");
        }
    }
    
    public function ask(string $question): string {
        return $this->chat->ask($question);
    }
    
    public function setSystemMessage(string $message): void {
        $this->chat->setSystemMessage($message);
    }
}

class LlamaEmbeddings {
    private LlamaEmbeddingInterface $embedder;
    
    public function __construct(string $provider = 'openai', string $model = null) {
        LlamaConfig::loadFromEnv();
        
        switch (strtolower($provider)) {
            case 'openai':
                if (!LlamaConfig::get('openai_api_key')) {
                    throw new Exception('OpenAI API key not set');
                }
                $this->embedder = new LlamaOpenAIEmbeddings($model);
                break;
            case 'ollama':
                $this->embedder = new LlamaOllamaEmbeddings($model);
                break;
            default:
                throw new Exception("Unsupported embedding provider: $provider");
        }
    }
    
    public function embed(string $text): array {
        return $this->embedder->embed($text);
    }
    
    public function embedBatch(array $texts): array {
        return $this->embedder->embedBatch($texts);
    }
    
    public function embedDocuments(array $documents): array {
        $texts = array_map(fn($doc) => $doc->content, $documents);
        $embeddings = $this->embedBatch($texts);
        
        foreach ($documents as $i => $doc) {
            $doc->embedding = $embeddings[$i] ?? [];
        }
        
        return $documents;
    }
}

class LlamaRAG {
    private LlamaVectorStoreInterface $vectorStore;
    private LlamaEmbeddingInterface $embedder;
    private LlamaChatInterface $chat;
    
    public function __construct(
        string $chatProvider = 'openai',
        string $embeddingProvider = 'openai',
        string $vectorStore = 'memory'
    ) {
        LlamaConfig::loadFromEnv();
        
        // Initialize vector store
        switch (strtolower($vectorStore)) {
            case 'memory':
                $this->vectorStore = new LlamaMemoryVectorStore();
                break;
            case 'file':
                $this->vectorStore = new LlamaFileVectorStore();
                break;
            default:
                throw new Exception("Unsupported vector store: $vectorStore");
        }
        
        // Initialize embedder
        $embeddings = new LlamaEmbeddings($embeddingProvider);
        $this->embedder = $embeddings->embedder;
        
        // Initialize chat
        $chat = new LlamaChat($chatProvider);
        $this->chat = $chat->chat;
    }
    
    public function addDocument(string $content, string $id = null, array $metadata = []): void {
        $doc = new LlamaDocument($content, $id, $metadata);
        $embedding = $this->embedder->embed($content);
        $this->vectorStore->add($doc->id, $embedding, $content, $metadata);
    }
    
    public function addDocuments(array $documents): void {
        foreach ($documents as $doc) {
            if (is_string($doc)) {
                $this->addDocument($doc);
            } elseif ($doc instanceof LlamaDocument) {
                $this->addDocument($doc->content, $doc->id, $doc->metadata);
            }
        }
    }
    
    public function loadText(string $text, int $chunkSize = 1000): void {
        $chunks = LlamaDocumentSplitter::split($text, $chunkSize);
        foreach ($chunks as $chunk) {
            $this->addDocument($chunk->content, $chunk->id);
        }
    }
    
    public function loadFile(string $filename, int $chunkSize = 1000): void {
        if (!file_exists($filename)) {
            throw new Exception("File not found: $filename");
        }
        
        $content = file_get_contents($filename);
        $this->loadText($content, $chunkSize);
    }
    
    public function ask(string $question, int $contextLimit = 5): string {
        // Get question embedding
        $questionEmbedding = $this->embedder->embed($question);
        
        // Search for relevant documents
        $results = $this->vectorStore->search($questionEmbedding, $contextLimit);
        
        if (empty($results)) {
            return $this->chat->ask($question);
        }
        
        // Build context from search results
        $context = "Context information:\n\n";
        foreach ($results as $result) {
            $context .= "- " . $result['content'] . "\n\n";
        }
        
        // Create enhanced prompt
        $prompt = $context . "\nQuestion: " . $question . "\n\nPlease answer the question based on the context information provided above.";
        
        return $this->chat->ask($prompt);
    }
    
    public function getRelevantDocuments(string $question, int $limit = 5): array {
        $questionEmbedding = $this->embedder->embed($question);
        return $this->vectorStore->search($questionEmbedding, $limit);
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function llama_chat(string $question, string $provider = 'openai'): string {
    static $instances = [];
    
    if (!isset($instances[$provider])) {
        $instances[$provider] = new LlamaChat($provider);
    }
    
    return $instances[$provider]->ask($question);
}

function llama_embed(string $text, string $provider = 'openai'): array {
    static $instances = [];
    
    if (!isset($instances[$provider])) {
        $instances[$provider] = new LlamaEmbeddings($provider);
    }
    
    return $instances[$provider]->embed($text);
}

function llama_similarity(array $a, array $b): float {
    $store = new LlamaMemoryVectorStore();
    return $store->cosineSimilarity($a, $b);
}

// =============================================================================
// EXAMPLE USAGE
// =============================================================================

/*
// Basic usage examples:

// 1. Simple chat
$chat = new LlamaChat('openai');
$chat->setSystemMessage("You are a helpful assistant.");
echo $chat->ask("What is PHP?");

// 2. Using function helper
echo llama_chat("What is the capital of France?");

// 3. Embeddings
$embedder = new LlamaEmbeddings('openai');
$vector = $embedder->embed("Hello world");
echo "Embedding dimension: " . count($vector);

// 4. RAG (Question Answering)
$rag = new LlamaRAG();
$rag->addDocument("PHP is a popular programming language for web development.");
$rag->addDocument("Laravel is a PHP framework.");
$rag->addDocument("Symfony is another popular PHP framework.");

echo $rag->ask("What frameworks are available for PHP?");

// 5. Load text file for RAG
$rag = new LlamaRAG();
$rag->loadFile('documentation.txt');
echo $rag->ask("How do I install this software?");

// 6. Using different providers
$ollamaChat = new LlamaChat('ollama');
echo $ollamaChat->ask("Explain machine learning");

// 7. Vector search
$vectorStore = new LlamaMemoryVectorStore();
$embedder = new LlamaEmbeddings();

$docs = [
    "PHP is a server-side scripting language",
    "Python is popular for machine learning",
    "JavaScript runs in browsers"
];

foreach ($docs as $i => $doc) {
    $vector = $embedder->embed($doc);
    $vectorStore->add("doc_$i", $vector, $doc);
}

$queryVector = $embedder->embed("web programming languages");
$results = $vectorStore->search($queryVector, 2);

foreach ($results as $result) {
    echo "Similarity: " . $result['similarity'] . " - " . $result['content'] . "\n";
}
*/