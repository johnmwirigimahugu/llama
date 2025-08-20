<?php
/**
 * Llama.php - Enterprise-Grade PHP AI Framework
 * 
 * Features:
 * - Multiple AI providers (OpenAI, Anthropic, Ollama, Google Gemini, Mistral, Cohere)
 * - Advanced RAG with reranking, multi-query, prompt injection detection
 * - Enterprise vector stores (PostgreSQL, Redis, Elasticsearch, SQLite)
 * - Function calling/tools, vision, audio processing
 * - Streaming responses, async processing, rate limiting
 * - Comprehensive logging, metrics, error handling
 * - Security features, input validation, sanitization
 * - Caching, connection pooling, load balancing
 * - Configuration management, health checks
 * - Evaluation framework, A/B testing
 * - Docker support, horizontal scaling
 * 
 * @version 2.0.0
 * @author AI Framework Team
 * @license MIT
 */

declare(strict_types=1);

// PHP 8.1+ required
if (version_compare(PHP_VERSION, '8.1.0', '<')) {
    throw new RuntimeException('Llama.php requires PHP 8.1 or higher');
}

// =============================================================================
// CORE EXCEPTIONS
// =============================================================================

class LlamaException extends Exception {}
class LlamaConfigException extends LlamaException {}
class LlamaValidationException extends LlamaException {}
class LlamaSecurityException extends LlamaException {}
class LlamaRateLimitException extends LlamaException {}
class LlamaTimeoutException extends LlamaException {}
class LlamaConnectionException extends LlamaException {}

// =============================================================================
// CORE INTERFACES
// =============================================================================

interface LlamaChatInterface {
    public function ask(string $question, array $options = []): string;
    public function askStream(string $question, callable $callback = null): Generator;
    public function setSystemMessage(string $message): void;
    public function addTool(LlamaFunctionTool $tool): void;
    public function getUsage(): array;
}

interface LlamaEmbeddingInterface {
    public function embed(string $text): array;
    public function embedBatch(array $texts): array;
    public function getDimension(): int;
}

interface LlamaVectorStoreInterface {
    public function add(string $id, array $vector, string $content, array $metadata = []): void;
    public function addBatch(array $documents): void;
    public function search(array $vector, int $limit = 5, array $filters = []): array;
    public function get(string $id): ?array;
    public function delete(string $id): bool;
    public function update(string $id, array $data): bool;
    public function count(): int;
    public function clear(): void;
    public function createIndex(array $options = []): bool;
}

interface LlamaCacheInterface {
    public function get(string $key): mixed;
    public function set(string $key, mixed $value, int $ttl = 3600): bool;
    public function delete(string $key): bool;
    public function clear(): bool;
    public function has(string $key): bool;
}

interface LlamaLoggerInterface {
    public function emergency(string $message, array $context = []): void;
    public function alert(string $message, array $context = []): void;
    public function critical(string $message, array $context = []): void;
    public function error(string $message, array $context = []): void;
    public function warning(string $message, array $context = []): void;
    public function notice(string $message, array $context = []): void;
    public function info(string $message, array $context = []): void;
    public function debug(string $message, array $context = []): void;
}

// =============================================================================
// CONFIGURATION & SECURITY
// =============================================================================

class LlamaConfig {
    private static array $config = [
        // API Configuration
        'api_keys' => [
            'openai' => null,
            'anthropic' => null,
            'google' => null,
            'mistral' => null,
            'cohere' => null,
        ],
        'api_urls' => [
            'ollama' => 'http://localhost:11434',
            'openai' => 'https://api.openai.com/v1',
            'anthropic' => 'https://api.anthropic.com/v1',
            'google' => 'https://generativelanguage.googleapis.com/v1',
        ],
        
        // Default Models
        'default_models' => [
            'openai' => 'gpt-4-turbo',
            'anthropic' => 'claude-3-sonnet-20240229',
            'ollama' => 'llama3',
            'google' => 'gemini-pro',
            'mistral' => 'mistral-large-latest',
        ],
        
        'embedding_models' => [
            'openai' => 'text-embedding-3-large',
            'ollama' => 'nomic-embed-text',
            'cohere' => 'embed-multilingual-v3.0',
        ],
        
        // Performance & Limits
        'timeouts' => [
            'default' => 30,
            'streaming' => 120,
            'embedding' => 60,
        ],
        'rate_limits' => [
            'requests_per_minute' => 100,
            'tokens_per_minute' => 100000,
        ],
        'batch_sizes' => [
            'embedding' => 100,
            'documents' => 50,
        ],
        
        // Security
        'security' => [
            'enable_input_validation' => true,
            'enable_prompt_injection_detection' => true,
            'max_input_length' => 100000,
            'allowed_file_types' => ['txt', 'pdf', 'docx', 'md', 'json', 'csv'],
            'enable_content_filtering' => true,
        ],
        
        // Logging & Monitoring
        'logging' => [
            'level' => 'INFO',
            'file' => '/tmp/llama.log',
            'max_size' => '100MB',
            'retention_days' => 30,
        ],
        'metrics' => [
            'enabled' => true,
            'endpoint' => '/metrics',
        ],
        
        // Caching
        'cache' => [
            'enabled' => true,
            'type' => 'file', // file, redis, memory
            'ttl' => 3600,
            'path' => '/tmp/llama_cache',
        ],
        
        // Vector Store
        'vector_store' => [
            'type' => 'memory', // memory, file, postgresql, redis, elasticsearch
            'connection' => [
                'host' => 'localhost',
                'port' => 5432,
                'database' => 'llama',
                'username' => 'llama',
                'password' => '',
            ],
        ],
    ];
    
    private static array $secrets = [];
    
    public static function load(array $config = []): void {
        self::$config = array_merge_recursive(self::$config, $config);
        self::loadFromEnv();
        self::validate();
    }
    
    public static function get(string $key, mixed $default = null): mixed {
        return self::getValue(self::$config, $key, $default);
    }
    
    public static function set(string $key, mixed $value): void {
        self::setValue(self::$config, $key, $value);
    }
    
    public static function setSecret(string $key, string $value): void {
        self::$secrets[$key] = $value;
    }
    
    public static function getSecret(string $key): ?string {
        return self::$secrets[$key] ?? null;
    }
    
    private static function loadFromEnv(): void {
        $envMappings = [
            'OPENAI_API_KEY' => 'api_keys.openai',
            'ANTHROPIC_API_KEY' => 'api_keys.anthropic',
            'GOOGLE_API_KEY' => 'api_keys.google',
            'MISTRAL_API_KEY' => 'api_keys.mistral',
            'COHERE_API_KEY' => 'api_keys.cohere',
            'LLAMA_LOG_LEVEL' => 'logging.level',
            'LLAMA_CACHE_TYPE' => 'cache.type',
            'POSTGRES_HOST' => 'vector_store.connection.host',
            'POSTGRES_PORT' => 'vector_store.connection.port',
            'POSTGRES_DB' => 'vector_store.connection.database',
            'POSTGRES_USER' => 'vector_store.connection.username',
            'POSTGRES_PASSWORD' => 'vector_store.connection.password',
        ];
        
        foreach ($envMappings as $env => $config) {
            $value = $_ENV[$env] ?? getenv($env);
            if ($value !== false) {
                self::setValue(self::$config, $config, $value);
            }
        }
    }
    
    private static function validate(): void {
        // Validate required settings
        $required = [
            'timeouts.default' => 'integer',
            'security.max_input_length' => 'integer',
            'rate_limits.requests_per_minute' => 'integer',
        ];
        
        foreach ($required as $key => $type) {
            $value = self::get($key);
            if ($value === null) {
                throw new LlamaConfigException("Required configuration missing: $key");
            }
            if (!self::validateType($value, $type)) {
                throw new LlamaConfigException("Invalid type for $key, expected $type");
            }
        }
    }
    
    private static function getValue(array $array, string $key, mixed $default = null): mixed {
        $keys = explode('.', $key);
        $current = $array;
        
        foreach ($keys as $k) {
            if (!is_array($current) || !array_key_exists($k, $current)) {
                return $default;
            }
            $current = $current[$k];
        }
        
        return $current;
    }
    
    private static function setValue(array &$array, string $key, mixed $value): void {
        $keys = explode('.', $key);
        $current = &$array;
        
        foreach ($keys as $k) {
            if (!is_array($current)) {
                $current = [];
            }
            if (!array_key_exists($k, $current)) {
                $current[$k] = [];
            }
            $current = &$current[$k];
        }
        
        $current = $value;
    }
    
    private static function validateType(mixed $value, string $type): bool {
        return match($type) {
            'string' => is_string($value),
            'integer' => is_int($value),
            'float' => is_float($value),
            'boolean' => is_bool($value),
            'array' => is_array($value),
            default => true,
        };
    }
}

class LlamaSecurity {
    private static array $bannedPatterns = [
        '/ignore\s+previous\s+instructions/i',
        '/you\s+are\s+now\s+a/i',
        '/jailbreak/i',
        '/prompt\s+injection/i',
        '/\bsql\s+injection\b/i',
        '/<script.*?>.*?<\/script>/is',
        '/javascript:/i',
    ];
    
    public static function validateInput(string $input): void {
        if (!LlamaConfig::get('security.enable_input_validation')) {
            return;
        }
        
        $maxLength = LlamaConfig::get('security.max_input_length');
        if (strlen($input) > $maxLength) {
            throw new LlamaValidationException("Input too long: " . strlen($input) . " > $maxLength");
        }
        
        if (LlamaConfig::get('security.enable_prompt_injection_detection')) {
            self::detectPromptInjection($input);
        }
    }
    
    public static function detectPromptInjection(string $input): void {
        foreach (self::$bannedPatterns as $pattern) {
            if (preg_match($pattern, $input)) {
                throw new LlamaSecurityException("Potential prompt injection detected");
            }
        }
    }
    
    public static function sanitizeInput(string $input): string {
        // Remove null bytes
        $input = str_replace("\0", '', $input);
        
        // Normalize whitespace
        $input = preg_replace('/\s+/', ' ', trim($input));
        
        // Remove potentially dangerous HTML
        $input = strip_tags($input);
        
        return $input;
    }
    
    public static function validateFileType(string $filename): void {
        $allowedTypes = LlamaConfig::get('security.allowed_file_types');
        $extension = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
        
        if (!in_array($extension, $allowedTypes)) {
            throw new LlamaSecurityException("File type not allowed: $extension");
        }
    }
}

// =============================================================================
// LOGGING & MONITORING
// =============================================================================

enum LlamaLogLevel: string {
    case EMERGENCY = 'EMERGENCY';
    case ALERT = 'ALERT';
    case CRITICAL = 'CRITICAL';
    case ERROR = 'ERROR';
    case WARNING = 'WARNING';
    case NOTICE = 'NOTICE';
    case INFO = 'INFO';
    case DEBUG = 'DEBUG';
}

class LlamaLogger implements LlamaLoggerInterface {
    private static ?self $instance = null;
    private string $logFile;
    private LlamaLogLevel $level;
    
    private function __construct() {
        $this->logFile = LlamaConfig::get('logging.file', '/tmp/llama.log');
        $this->level = LlamaLogLevel::from(LlamaConfig::get('logging.level', 'INFO'));
    }
    
    public static function getInstance(): self {
        if (self::$instance === null) {
            self::$instance = new self();
        }
        return self::$instance;
    }
    
    public function emergency(string $message, array $context = []): void {
        $this->log(LlamaLogLevel::EMERGENCY, $message, $context);
    }
    
    public function alert(string $message, array $context = []): void {
        $this->log(LlamaLogLevel::ALERT, $message, $context);
    }
    
    public function critical(string $message, array $context = []): void {
        $this->log(LlamaLogLevel::CRITICAL, $message, $context);
    }
    
    public function error(string $message, array $context = []): void {
        $this->log(LlamaLogLevel::ERROR, $message, $context);
    }
    
    public function warning(string $message, array $context = []): void {
        $this->log(LlamaLogLevel::WARNING, $message, $context);
    }
    
    public function notice(string $message, array $context = []): void {
        $this->log(LlamaLogLevel::NOTICE, $message, $context);
    }
    
    public function info(string $message, array $context = []): void {
        $this->log(LlamaLogLevel::INFO, $message, $context);
    }
    
    public function debug(string $message, array $context = []): void {
        $this->log(LlamaLogLevel::DEBUG, $message, $context);
    }
    
    private function log(LlamaLogLevel $level, string $message, array $context = []): void {
        if (!$this->shouldLog($level)) {
            return;
        }
        
        $timestamp = date('Y-m-d H:i:s');
        $contextStr = empty($context) ? '' : ' ' . json_encode($context);
        $logEntry = "[$timestamp] {$level->value}: $message$contextStr\n";
        
        // Ensure directory exists
        $dir = dirname($this->logFile);
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }
        
        file_put_contents($this->logFile, $logEntry, FILE_APPEND | LOCK_EX);
        
        // Rotate logs if needed
        $this->rotateLogsIfNeeded();
    }
    
    private function shouldLog(LlamaLogLevel $level): bool {
        $levels = [
            LlamaLogLevel::EMERGENCY->value => 0,
            LlamaLogLevel::ALERT->value => 1,
            LlamaLogLevel::CRITICAL->value => 2,
            LlamaLogLevel::ERROR->value => 3,
            LlamaLogLevel::WARNING->value => 4,
            LlamaLogLevel::NOTICE->value => 5,
            LlamaLogLevel::INFO->value => 6,
            LlamaLogLevel::DEBUG->value => 7,
        ];
        
        return $levels[$level->value] <= $levels[$this->level->value];
    }
    
    private function rotateLogsIfNeeded(): void {
        if (!file_exists($this->logFile)) {
            return;
        }
        
        $maxSize = $this->parseSize(LlamaConfig::get('logging.max_size', '100MB'));
        if (filesize($this->logFile) > $maxSize) {
            $rotated = $this->logFile . '.' . date('Y-m-d-H-i-s');
            rename($this->logFile, $rotated);
            
            // Compress old log
            if (function_exists('gzencode')) {
                $content = file_get_contents($rotated);
                file_put_contents($rotated . '.gz', gzencode($content));
                unlink($rotated);
            }
        }
        
        $this->cleanOldLogs();
    }
    
    private function cleanOldLogs(): void {
        $retentionDays = LlamaConfig::get('logging.retention_days', 30);
        $cutoff = time() - ($retentionDays * 24 * 60 * 60);
        $dir = dirname($this->logFile);
        $pattern = basename($this->logFile) . '.*';
        
        foreach (glob("$dir/$pattern") as $file) {
            if (filemtime($file) < $cutoff) {
                unlink($file);
            }
        }
    }
    
    private function parseSize(string $size): int {
        $size = strtoupper($size);
        $multipliers = ['B' => 1, 'KB' => 1024, 'MB' => 1024**2, 'GB' => 1024**3];
        
        if (preg_match('/^(\d+)([A-Z]*)$/', $size, $matches)) {
            $value = (int)$matches[1];
            $unit = $matches[2] ?: 'B';
            return $value * ($multipliers[$unit] ?? 1);
        }
        
        return 100 * 1024 * 1024; // Default 100MB
    }
}

class LlamaMetrics {
    private static array $counters = [];
    private static array $timers = [];
    private static array $gauges = [];
    
    public static function increment(string $name, int $value = 1, array $tags = []): void {
        $key = self::buildKey($name, $tags);
        self::$counters[$key] = (self::$counters[$key] ?? 0) + $value;
    }
    
    public static function timer(string $name, callable $callback, array $tags = []): mixed {
        $start = microtime(true);
        try {
            $result = $callback();
            $duration = microtime(true) - $start;
            self::timing($name, $duration * 1000, $tags); // Convert to milliseconds
            return $result;
        } catch (Exception $e) {
            $duration = microtime(true) - $start;
            self::timing($name, $duration * 1000, array_merge($tags, ['error' => true]));
            throw $e;
        }
    }
    
    public static function timing(string $name, float $time, array $tags = []): void {
        $key = self::buildKey($name, $tags);
        if (!isset(self::$timers[$key])) {
            self::$timers[$key] = [];
        }
        self::$timers[$key][] = $time;
    }
    
    public static function gauge(string $name, float $value, array $tags = []): void {
        $key = self::buildKey($name, $tags);
        self::$gauges[$key] = $value;
    }
    
    public static function getMetrics(): array {
        return [
            'counters' => self::$counters,
            'timers' => array_map(function($times) {
                return [
                    'count' => count($times),
                    'min' => min($times),
                    'max' => max($times),
                    'avg' => array_sum($times) / count($times),
                    'sum' => array_sum($times),
                ];
            }, self::$timers),
            'gauges' => self::$gauges,
            'timestamp' => time(),
        ];
    }
    
    private static function buildKey(string $name, array $tags): string {
        if (empty($tags)) {
            return $name;
        }
        ksort($tags);
        $tagStr = implode(',', array_map(fn($k, $v) => "$k=$v", array_keys($tags), $tags));
        return "$name{$tagStr}";
    }
}

// =============================================================================
// CACHING
// =============================================================================

class LlamaFileCache implements LlamaCacheInterface {
    private string $cacheDir;
    
    public function __construct(string $cacheDir = null) {
        $this->cacheDir = $cacheDir ?? LlamaConfig::get('cache.path', sys_get_temp_dir() . '/llama_cache');
        if (!is_dir($this->cacheDir)) {
            mkdir($this->cacheDir, 0755, true);
        }
    }
    
    public function get(string $key): mixed {
        $file = $this->getFilePath($key);
        if (!file_exists($file)) {
            return null;
        }
        
        $data = unserialize(file_get_contents($file));
        if ($data['expires'] > 0 && $data['expires'] < time()) {
            $this->delete($key);
            return null;
        }
        
        return $data['value'];
    }
    
    public function set(string $key, mixed $value, int $ttl = 3600): bool {
        $file = $this->getFilePath($key);
        $data = [
            'value' => $value,
            'expires' => $ttl > 0 ? time() + $ttl : 0,
        ];
        
        return file_put_contents($file, serialize($data), LOCK_EX) !== false;
    }
    
    public function delete(string $key): bool {
        $file = $this->getFilePath($key);
        return !file_exists($file) || unlink($file);
    }
    
    public function clear(): bool {
        $files = glob($this->cacheDir . '/*');
        foreach ($files as $file) {
            if (is_file($file)) {
                unlink($file);
            }
        }
        return true;
    }
    
    public function has(string $key): bool {
        return $this->get($key) !== null;
    }
    
    private function getFilePath(string $key): string {
        $hash = hash('sha256', $key);
        return $this->cacheDir . '/' . $hash . '.cache';
    }
}

class LlamaMemoryCache implements LlamaCacheInterface {
    private static array $cache = [];
    private static array $expires = [];
    
    public function get(string $key): mixed {
        if (!isset(self::$cache[$key])) {
            return null;
        }
        
        if (isset(self::$expires[$key]) && self::$expires[$key] < time()) {
            $this->delete($key);
            return null;
        }
        
        return self::$cache[$key];
    }
    
    public function set(string $key, mixed $value, int $ttl = 3600): bool {
        self::$cache[$key] = $value;
        if ($ttl > 0) {
            self::$expires[$key] = time() + $ttl;
        }
        return true;
    }
    
    public function delete(string $key): bool {
        unset(self::$cache[$key], self::$expires[$key]);
        return true;
    }
    
    public function clear(): bool {
        self::$cache = [];
        self::$expires = [];
        return true;
    }
    
    public function has(string $key): bool {
        return $this->get($key) !== null;
    }
}

// =============================================================================
// RATE LIMITING
// =============================================================================

class LlamaRateLimiter {
    private LlamaCacheInterface $cache;
    private int $requestsPerMinute;
    private int $tokensPerMinute;
    
    public function __construct(LlamaCacheInterface $cache = null) {
        $this->cache = $cache ?? new LlamaMemoryCache();
        $this->requestsPerMinute = LlamaConfig::get('rate_limits.requests_per_minute', 100);
        $this->tokensPerMinute = LlamaConfig::get('rate_limits.tokens_per_minute', 100000);
    }
    
    public function checkRequest(string $identifier): void {
        $window = floor(time() / 60); // 1-minute windows
        $key = "requests:$identifier:$window";
        
        $current = $this->cache->get($key) ?? 0;
        if ($current >= $this->requestsPerMinute) {
            throw new LlamaRateLimitException("Request rate limit exceeded: $current/$this->requestsPerMinute per minute");
        }
        
        $this->cache->set($key, $current + 1, 120); // Keep for 2 minutes
    }
    
    public function checkTokens(string $identifier, int $tokens): void {
        $window = floor(time() / 60);
        $key = "tokens:$identifier:$window";
        
        $current = $this->cache->get($key) ?? 0;
        if ($current + $tokens > $this->tokensPerMinute) {
            throw new LlamaRateLimitException("Token rate limit exceeded: " . ($current + $tokens) . "/$this->tokensPerMinute per minute");
        }
        
        $this->cache->set($key, $current + $tokens, 120);
    }
}

// =============================================================================
// HTTP CLIENT WITH ADVANCED FEATURES
// =============================================================================

class LlamaHttpClient {
    private int $timeout;
    private int $connectTimeout;
    private int $maxRetries;
    private LlamaCacheInterface $cache;
    private LlamaRateLimiter $rateLimiter;
    private LlamaLoggerInterface $logger;
    
    public function __construct(array $options = []) {
        $this->timeout = $options['timeout'] ?? LlamaConfig::get('timeouts.default', 30);
        $this->connectTimeout = $options['connect_timeout'] ?? 10;
        $this->maxRetries = $options['max_retries'] ?? 3;
        $this->cache = $options['cache'] ?? new LlamaMemoryCache();
        $this->rateLimiter = $options['rate_limiter'] ?? new LlamaRateLimiter();
        $this->logger = $options['logger'] ?? LlamaLogger::getInstance();
    }
    
    public function post(string $url, array $data, array $headers = [], array $options = []): array {
        return $this->request('POST', $url, $data, $headers, $options);
    }
    
    public function get(string $url, array $headers = [], array $options = []): array {
        return $this->request('GET', $url, null, $headers, $options);
    }
    
    public function postStream(string $url, array $data, array $headers = [], callable $callback = null): Generator {
        $cacheKey = $options['cache_key'] ?? null;
        if ($cacheKey && $this->cache->has($cacheKey)) {
            $cached = $this->cache->get($cacheKey);
            foreach ($cached as $chunk) {
                yield $chunk;
            }
            return;
        }
        
        $ch = $this->createCurlHandle('POST', $url, $data, $headers, ['stream' => true]);
        
        $chunks = [];
        curl_setopt($ch, CURLOPT_WRITEFUNCTION, function($ch, $data) use ($callback, &$chunks) {
            $chunks[] = $data;
            if ($callback) {
                $callback($data);
            }
            return strlen($data);
        });
        
        $this->logger->debug("Starting streaming request to $url");
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        if ($error) {
            throw new LlamaConnectionException("Stream error: $error");
        }
        
        if ($httpCode >= 400) {
            throw new LlamaException("Stream HTTP error $httpCode");
        }
        
        // Cache stream data if requested
        if ($cacheKey) {
            $this->cache->set($cacheKey, $chunks, 300); // 5 minutes
        }
        
        foreach ($chunks as $chunk) {
            yield $chunk;
        }
    }
    
    private function request(string $method, string $url, mixed $data = null, array $headers = [], array $options = []): array {
        $cacheKey = $options['cache_key'] ?? null;
        if ($cacheKey && $this->cache->has($cacheKey)) {
            return $this->cache->get($cacheKey);
        }
        
        // Rate limiting
        if (!($options['skip_rate_limit'] ?? false)) {
            $identifier = $options['rate_limit_key'] ?? 'default';
            $this->rateLimiter->checkRequest($identifier);
        }
        
        $attempt = 0;
        $lastException = null;
        
        while ($attempt < $this->maxRetries) {
            try {
                $result = $this->executeRequest($method, $url, $data, $headers, $options);
                
                // Cache successful responses
                if ($cacheKey && $result) {
                    $ttl = $options['cache_ttl'] ?? 300;
                    $this->cache->set($cacheKey, $result, $ttl);
                }
                
                return $result;
                
            } catch (LlamaTimeoutException|LlamaConnectionException $e) {
                $lastException = $e;
                $attempt++;
                if ($attempt < $this->maxRetries) {
                    $delay = pow(2, $attempt - 1); // Exponential backoff
                $this->logger->warning("Request failed, retrying in {$delay}s", ['attempt' => $attempt, 'url' => $url]);
                sleep($delay);
            }
            } catch (Exception $e) {
                throw $e; // Don't retry other exceptions
            }
        }
        
        throw $lastException ?? new LlamaException("Request failed after $this->maxRetries attempts");
    }
    
    private function executeRequest(string $method, string $url, mixed $data, array $headers, array $options): array {
        $ch = $this->createCurlHandle($method, $url, $data, $headers, $options);
        
        $startTime = microtime(true);
        $response = curl_exec($ch);
        $duration = microtime(true) - $startTime;
        
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        // Log request metrics
        LlamaMetrics::timing('http.request.duration', $duration * 1000, ['method' => $method]);
        LlamaMetrics::increment('http.request.count', 1, ['method' => $method, 'status' => $httpCode]);
        
        if ($error) {
            if (strpos($error, 'timeout') !== false) {
                throw new LlamaTimeoutException("Request timeout: $error");
            }
            throw new LlamaConnectionException("HTTP Error: $error");
        }
        
        if ($httpCode >= 400) {
            $this->logger->error("HTTP error", ['code' => $httpCode, 'response' => $response, 'url' => $url]);
            throw new LlamaException("HTTP Error $httpCode: $response");
        }
        
        $decoded = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new LlamaException("JSON decode error: " . json_last_error_msg());
        }
        
        return $decoded;
    }
    
    private function createCurlHandle(string $method, string $url, mixed $data, array $headers, array $options): \CurlHandle {
        $ch = curl_init();
        
        $curlOptions = [
            CURLOPT_URL => $url,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $options['timeout'] ?? $this->timeout,
            CURLOPT_CONNECTTIMEOUT => $this->connectTimeout,
            CURLOPT_FOLLOWLOCATION => true,
            CURLOPT_MAXREDIRS => 3,
            CURLOPT_SSL_VERIFYPEER => true,
            CURLOPT_SSL_VERIFYHOST => 2,
            CURLOPT_USERAGENT => 'Llama-PHP/2.0',
            CURLOPT_HTTPHEADER => array_merge(['Content-Type: application/json'], $headers),
        ];
        
        if ($method === 'POST') {
            $curlOptions[CURLOPT_POST] = true;
            if ($data !== null) {
                $curlOptions[CURLOPT_POSTFIELDS] = is_string($data) ? $data : json_encode($data);
            }
        }
        
        if ($options['stream'] ?? false) {
            $curlOptions[CURLOPT_TIMEOUT] = LlamaConfig::get('timeouts.streaming', 120);
        }
        
        curl_setopt_array($ch, $curlOptions);
        return $ch;
    }
}

// =============================================================================
// ADVANCED DOCUMENT PROCESSING
// =============================================================================

class LlamaDocument {
    public string $id;
    public string $content;
    public array $metadata;
    public ?array $embedding = null;
    public ?string $sourceType = null;
    public ?string $sourceName = null;
    public int $tokens = 0;
    public \DateTime $createdAt;
    public \DateTime $updatedAt;
    
    public function __construct(string $content, string $id = null, array $metadata = []) {
        $this->id = $id ?? 'doc_' . uniqid();
        $this->content = $content;
        $this->metadata = $metadata;
        $this->tokens = $this->estimateTokens($content);
        $this->createdAt = new \DateTime();
        $this->updatedAt = new \DateTime();
        
        // Extract metadata
        $this->sourceType = $metadata['source_type'] ?? 'text';
        $this->sourceName = $metadata['source_name'] ?? null;
    }
    
    private function estimateTokens(string $content): int {
        // Rough token estimation (1 token ≈ 4 characters for English)
        return (int)ceil(strlen($content) / 4);
    }
    
    public function toArray(): array {
        return [
            'id' => $this->id,
            'content' => $this->content,
            'metadata' => $this->metadata,
            'embedding' => $this->embedding,
            'tokens' => $this->tokens,
            'created_at' => $this->createdAt->format('Y-m-d H:i:s'),
            'updated_at' => $this->updatedAt->format('Y-m-d H:i:s'),
        ];
    }
}

class LlamaDocumentProcessor {
    private LlamaLoggerInterface $logger;
    
    public function __construct() {
        $this->logger = LlamaLogger::getInstance();
    }
    
    public function loadFile(string $filename): array {
        LlamaSecurity::validateFileType($filename);
        
        if (!file_exists($filename)) {
            throw new LlamaException("File not found: $filename");
        }
        
        $extension = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
        $content = match($extension) {
            'txt', 'md' => file_get_contents($filename),
            'json' => $this->processJson($filename),
            'csv' => $this->processCsv($filename),
            'pdf' => $this->processPdf($filename),
            'docx' => $this->processDocx($filename),
            default => throw new LlamaException("Unsupported file type: $extension"),
        };
        
        return [new LlamaDocument($content, basename($filename, ".$extension"), [
            'source_type' => 'file',
            'source_name' => $filename,
            'file_extension' => $extension,
            'file_size' => filesize($filename),
        ])];
    }
    
    public function loadDirectory(string $directory, bool $recursive = true): array {
        if (!is_dir($directory)) {
            throw new LlamaException("Directory not found: $directory");
        }
        
        $documents = [];
        $iterator = $recursive 
            ? new \RecursiveIteratorIterator(new \RecursiveDirectoryIterator($directory))
            : new \DirectoryIterator($directory);
        
        foreach ($iterator as $file) {
            if ($file->isFile()) {
                try {
                    $docs = $this->loadFile($file->getPathname());
                    $documents = array_merge($documents, $docs);
                } catch (Exception $e) {
                    $this->logger->warning("Failed to process file: " . $file->getPathname(), ['error' => $e->getMessage()]);
                }
            }
        }
        
        return $documents;
    }
    
    public function loadUrl(string $url): array {
        $client = new LlamaHttpClient();
        
        if (filter_var($url, FILTER_VALIDATE_URL) === false) {
            throw new LlamaValidationException("Invalid URL: $url");
        }
        
        // Simple HTML content extraction
        $html = $client->get($url);
        $content = $this->extractTextFromHtml($html['content'] ?? '');
        
        return [new LlamaDocument($content, md5($url), [
            'source_type' => 'url',
            'source_name' => $url,
            'content_type' => 'html',
        ])];
    }
    
    private function processJson(string $filename): string {
        $data = json_decode(file_get_contents($filename), true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new LlamaException("Invalid JSON file: $filename");
        }
        
        return $this->flattenData($data);
    }
    
    private function processCsv(string $filename): string {
        $handle = fopen($filename, 'r');
        if (!$handle) {
            throw new LlamaException("Cannot open CSV file: $filename");
        }
        
        $content = '';
        $header = fgetcsv($handle);
        
        while (($row = fgetcsv($handle)) !== false) {
            if ($header && count($header) === count($row)) {
                $assoc = array_combine($header, $row);
                $content .= implode(': ', array_map(fn($k, $v) => "$k=$v", array_keys($assoc), $assoc)) . "\n";
            }
        }
        
        fclose($handle);
        return $content;
    }
    
    private function processPdf(string $filename): string {
        // Basic PDF text extraction (requires pdftotext or similar)
        if (command_exists('pdftotext')) {
            $tempFile = tempnam(sys_get_temp_dir(), 'llama_pdf_');
            exec("pdftotext '$filename' '$tempFile' 2>/dev/null", $output, $returnCode);
            
            if ($returnCode === 0 && file_exists($tempFile)) {
                $content = file_get_contents($tempFile);
                unlink($tempFile);
                return $content;
            }
        }
        
        throw new LlamaException("PDF processing not available. Install pdftotext or provide plain text.");
    }
    
    private function processDocx(string $filename): string {
        // Basic DOCX processing (extract from document.xml)
        if (!class_exists('ZipArchive')) {
            throw new LlamaException("ZipArchive required for DOCX processing");
        }
        
        $zip = new \ZipArchive();
        if ($zip->open($filename) !== true) {
            throw new LlamaException("Cannot open DOCX file: $filename");
        }
        
        $xml = $zip->getFromName('word/document.xml');
        $zip->close();
        
        if ($xml === false) {
            throw new LlamaException("Cannot extract content from DOCX");
        }
        
        // Simple XML text extraction
        $dom = new \DOMDocument();
        $dom->loadXML($xml);
        return $dom->textContent;
    }
    
    private function extractTextFromHtml(string $html): string {
        // Remove scripts and styles
        $html = preg_replace('/<(script|style)[^>]*>.*?<\/\1>/is', '', $html);
        
        // Convert to plain text
        $text = strip_tags($html);
        
        // Clean up whitespace
        $text = preg_replace('/\s+/', ' ', $text);
        
        return trim($text);
    }
    
    private function flattenData(array $data, string $prefix = ''): string {
        $result = '';
        
        foreach ($data as $key => $value) {
            $newKey = $prefix ? "$prefix.$key" : $key;
            
            if (is_array($value)) {
                $result .= $this->flattenData($value, $newKey);
            } else {
                $result .= "$newKey: $value\n";
            }
        }
        
        return $result;
    }
}

class LlamaDocumentSplitter {
    public const STRATEGY_TOKENS = 'tokens';
    public const STRATEGY_SENTENCES = 'sentences';
    public const STRATEGY_PARAGRAPHS = 'paragraphs';
    public const STRATEGY_SEMANTIC = 'semantic';
    
    public static function split(
        string $text, 
        int $chunkSize = 1000, 
        int $overlap = 200, 
        string $strategy = self::STRATEGY_TOKENS
    ): array {
        return match($strategy) {
            self::STRATEGY_TOKENS => self::splitByTokens($text, $chunkSize, $overlap),
            self::STRATEGY_SENTENCES => self::splitBySentences($text, $chunkSize, $overlap),
            self::STRATEGY_PARAGRAPHS => self::splitByParagraphs($text, $chunkSize, $overlap),
            self::STRATEGY_SEMANTIC => self::splitSemantically($text, $chunkSize, $overlap),
            default => throw new LlamaException("Unknown splitting strategy: $strategy"),
        };
    }
    
    private static function splitByTokens(string $text, int $chunkSize, int $overlap): array {
        $words = explode(' ', $text);
        $chunks = [];
        
        for ($i = 0; $i < count($words); $i += ($chunkSize - $overlap)) {
            $chunk = array_slice($words, $i, $chunkSize);
            $chunkText = implode(' ', $chunk);
            
            if (!empty(trim($chunkText))) {
                $chunks[] = new LlamaDocument($chunkText, 'chunk_' . uniqid());
            }
            
            if ($i + $chunkSize >= count($words)) {
                break;
            }
        }
        
        return $chunks;
    }
    
    private static function splitBySentences(string $text, int $chunkSize, int $overlap): array {
        $sentences = preg_split('/(?<=[.!?])\s+/', $text, -1, PREG_SPLIT_NO_EMPTY);
        $chunks = [];
        $currentChunk = [];
        $currentSize = 0;
        
        foreach ($sentences as $sentence) {
            $sentenceSize = str_word_count($sentence);
            
            if ($currentSize + $sentenceSize > $chunkSize && !empty($currentChunk)) {
                $chunks[] = new LlamaDocument(implode(' ', $currentChunk), 'chunk_' . uniqid());
                
                // Handle overlap
                $overlapSentences = [];
                $overlapSize = 0;
                for ($i = count($currentChunk) - 1; $i >= 0 && $overlapSize < $overlap; $i--) {
                    $overlapSentences[] = $currentChunk[$i];
                    $overlapSize += str_word_count($currentChunk[$i]);
                }
                
                $currentChunk = array_reverse($overlapSentences);
                $currentSize = $overlapSize;
            }
            
            $currentChunk[] = $sentence;
            $currentSize += $sentenceSize;
        }
        
        if (!empty($currentChunk)) {
            $chunks[] = new LlamaDocument(implode(' ', $currentChunk), 'chunk_' . uniqid());
        }
        
        return $chunks;
    }
    
    private static function splitByParagraphs(string $text, int $chunkSize, int $overlap): array {
        $paragraphs = preg_split('/\n\s*\n/', $text, -1, PREG_SPLIT_NO_EMPTY);
        $chunks = [];
        $currentChunk = '';
        
        foreach ($paragraphs as $paragraph) {
            $paragraphSize = str_word_count($paragraph);
            
            if (str_word_count($currentChunk) + $paragraphSize > $chunkSize && !empty($currentChunk)) {
                $chunks[] = new LlamaDocument(trim($currentChunk), 'chunk_' . uniqid());
                
                // Simple overlap by taking last part of current chunk
                $words = explode(' ', $currentChunk);
                $overlapWords = array_slice($words, -$overlap);
                $currentChunk = implode(' ', $overlapWords) . ' ';
            }
            
            $currentChunk .= $paragraph . "\n\n";
        }
        
        if (!empty(trim($currentChunk))) {
            $chunks[] = new LlamaDocument(trim($currentChunk), 'chunk_' . uniqid());
        }
        
        return $chunks;
    }
    
    private static function splitSemantically(string $text, int $chunkSize, int $overlap): array {
        // Fallback to sentence-based splitting for now
        // In a full implementation, this would use embeddings to find semantic boundaries
        return self::splitBySentences($text, $chunkSize, $overlap);
    }
}

// =============================================================================
// FUNCTION CALLING / TOOLS
// =============================================================================

class LlamaFunctionParameter {
    public function __construct(
        public string $name,
        public string $type,
        public string $description,
        public bool $required = true,
        public mixed $default = null,
        public ?array $enum = null
    ) {}
    
    public function toArray(): array {
        $schema = [
            'type' => $this->type,
            'description' => $this->description,
        ];
        
        if ($this->enum !== null) {
            $schema['enum'] = $this->enum;
        }
        
        if ($this->default !== null) {
            $schema['default'] = $this->default;
        }
        
        return $schema;
    }
}

class LlamaFunctionTool {
    public function __construct(
        public string $name,
        public string $description,
        public array $parameters,
        public callable $function,
        public array $metadata = []
    ) {}
    
    public function call(array $arguments): mixed {
        // Validate arguments
        foreach ($this->parameters as $param) {
            if ($param->required && !isset($arguments[$param->name])) {
                throw new LlamaValidationException("Missing required parameter: {$param->name}");
            }
        }
        
        try {
            return call_user_func($this->function, $arguments);
        } catch (Exception $e) {
            throw new LlamaException("Tool execution failed: " . $e->getMessage(), 0, $e);
        }
    }
    
    public function getSchema(): array {
        $properties = [];
        $required = [];
        
        foreach ($this->parameters as $param) {
            $properties[$param->name] = $param->toArray();
            if ($param->required) {
                $required[] = $param->name;
            }
        }
        
        return [
            'type' => 'function',
            'function' => [
                'name' => $this->name,
                'description' => $this->description,
                'parameters' => [
                    'type' => 'object',
                    'properties' => $properties,
                    'required' => $required,
                ],
            ],
        ];
    }
}

// =============================================================================
// CHAT IMPLEMENTATIONS WITH ADVANCED FEATURES
// =============================================================================

abstract class LlamaBaseChatProvider implements LlamaChatInterface {
    protected LlamaHttpClient $client;
    protected string $model;
    protected ?string $systemMessage = null;
    protected array $messages = [];
    protected array $tools = [];
    protected array $usage = ['total_tokens' => 0, 'requests' => 0];
    protected LlamaLoggerInterface $logger;
    protected LlamaCacheInterface $cache;
    
    public function __construct(string $model = null) {
        $this->model = $model ?? $this->getDefaultModel();
        $this->client = new LlamaHttpClient();
        $this->logger = LlamaLogger::getInstance();
        $this->cache = LlamaConfig::get('cache.enabled') ? new LlamaFileCache() : new LlamaMemoryCache();
    }
    
    abstract protected function getDefaultModel(): string;
    abstract protected function buildRequest(string $question, array $options): array;
    abstract protected function extractResponse(array $response): string;
    abstract protected function getApiUrl(): string;
    abstract protected function getHeaders(): array;
    
    public function setSystemMessage(string $message): void {
        LlamaSecurity::validateInput($message);
        $this->systemMessage = LlamaSecurity::sanitizeInput($message);
    }
    
    public function addTool(LlamaFunctionTool $tool): void {
        $this->tools[$tool->name] = $tool;
        $this->logger->info("Added tool: {$tool->name}");
    }
    
    public function ask(string $question, array $options = []): string {
        LlamaSecurity::validateInput($question);
        $question = LlamaSecurity::sanitizeInput($question);
        
        return LlamaMetrics::timer('chat.ask', function() use ($question, $options) {
            $cacheKey = $options['cache'] ?? null;
            if ($cacheKey) {
                $cacheKey = 'chat:' . md5($this->model . $this->systemMessage . $question . serialize($options));
                if ($this->cache->has($cacheKey)) {
                    return $this->cache->get($cacheKey);
                }
            }
            
            $request = $this->buildRequest($question, $options);
            $response = $this->client->post($this->getApiUrl(), $request, $this->getHeaders(), [
                'rate_limit_key' => $options['rate_limit_key'] ?? 'default',
                'timeout' => $options['timeout'] ?? null,
            ]);
            
            $answer = $this->extractResponse($response);
            $this->updateUsage($response);
            
            // Handle tool calls if present
            if ($this->hasToolCalls($response)) {
                $answer = $this->handleToolCalls($response, $question, $options);
            }
            
            // Store in conversation history
            $this->messages[] = ['role' => 'user', 'content' => $question];
            $this->messages[] = ['role' => 'assistant', 'content' => $answer];
            
            // Cache response if requested
            if ($cacheKey) {
                $this->cache->set($cacheKey, $answer, $options['cache_ttl'] ?? 300);
            }
            
            LlamaMetrics::increment('chat.requests', 1, ['model' => $this->model]);
            return $answer;
        });
    }
    
    public function askStream(string $question, callable $callback = null): Generator {
        LlamaSecurity::validateInput($question);
        $question = LlamaSecurity::sanitizeInput($question);
        
        $request = $this->buildRequest($question, ['stream' => true]);
        $stream = $this->client->postStream($this->getApiUrl(), $request, $this->getHeaders());
        
        $fullResponse = '';
        foreach ($stream as $chunk) {
            $data = $this->parseStreamChunk($chunk);
            if ($data) {
                $fullResponse .= $data;
                if ($callback) {
                    $callback($data);
                }
                yield $data;
            }
        }
        
        // Store in conversation history
        $this->messages[] = ['role' => 'user', 'content' => $question];
        $this->messages[] = ['role' => 'assistant', 'content' => $fullResponse];
    }
    
    public function getUsage(): array {
        return $this->usage;
    }
    
    protected function hasToolCalls(array $response): bool {
        return false; // Override in specific implementations
    }
    
    protected function handleToolCalls(array $response, string $originalQuestion, array $options): string {
        return $this->extractResponse($response); // Override in specific implementations
    }
    
    protected function parseStreamChunk(string $chunk): ?string {
        // Parse SSE format: data: {...}
        if (strpos($chunk, 'data: ') === 0) {
            $json = substr($chunk, 6);
            if ($json === '[DONE]') {
                return null;
            }
            
            $data = json_decode(trim($json), true);
            if (json_last_error() === JSON_ERROR_NONE) {
                return $this->extractStreamContent($data);
            }
        }
        
        return null;
    }
    
    protected function extractStreamContent(array $data): ?string {
        return null; // Override in specific implementations
    }
    
    protected function updateUsage(array $response): void {
        if (isset($response['usage'])) {
            $this->usage['total_tokens'] += $response['usage']['total_tokens'] ?? 0;
        }
        $this->usage['requests']++;
    }
}

class LlamaOpenAIChat extends LlamaBaseChatProvider {
    protected function getDefaultModel(): string {
        return LlamaConfig::get('default_models.openai', 'gpt-4-turbo');
    }
    
    protected function getApiUrl(): string {
        return LlamaConfig::get('api_urls.openai') . '/chat/completions';
    }
    
    protected function getHeaders(): array {
        $apiKey = LlamaConfig::get('api_keys.openai');
        if (!$apiKey) {
            throw new LlamaConfigException('OpenAI API key not configured');
        }
        
        return ['Authorization: Bearer ' . $apiKey];
    }
    
    protected function buildRequest(string $question, array $options): array {
        $messages = [];
        
        if ($this->systemMessage) {
            $messages[] = ['role' => 'system', 'content' => $this->systemMessage];
        }
        
        $messages = array_merge($messages, $this->messages);
        $messages[] = ['role' => 'user', 'content' => $question];
        
        $request = [
            'model' => $this->model,
            'messages' => $messages,
            'temperature' => $options['temperature'] ?? 0.7,
            'max_tokens' => $options['max_tokens'] ?? 2000,
        ];
        
        if (!empty($this->tools)) {
            $request['tools'] = array_map(fn($tool) => $tool->getSchema(), $this->tools);
            $request['tool_choice'] = $options['tool_choice'] ?? 'auto';
        }
        
        if ($options['stream'] ?? false) {
            $request['stream'] = true;
        }
        
        return $request;
    }
    
    protected function extractResponse(array $response): string {
        return $response['choices'][0]['message']['content'] ?? '';
    }
    
    protected function extractStreamContent(array $data): ?string {
        return $data['choices'][0]['delta']['content'] ?? null;
    }
    
    protected function hasToolCalls(array $response): bool {
        return isset($response['choices'][0]['message']['tool_calls']);
    }
    
    protected function handleToolCalls(array $response, string $originalQuestion, array $options): string {
        $toolCalls = $response['choices'][0]['message']['tool_calls'];
        $results = [];
        
        foreach ($toolCalls as $toolCall) {
            $toolName = $toolCall['function']['name'];
            $arguments = json_decode($toolCall['function']['arguments'], true);
            
            if (isset($this->tools[$toolName])) {
                try {
                    $result = $this->tools[$toolName]->call($arguments);
                    $results[] = "Tool $toolName result: " . json_encode($result);
                    $this->logger->info("Tool executed", ['tool' => $toolName, 'arguments' => $arguments]);
                } catch (Exception $e) {
                    $results[] = "Tool $toolName error: " . $e->getMessage();
                    $this->logger->error("Tool execution failed", ['tool' => $toolName, 'error' => $e->getMessage()]);
                }
            }
        }
        
        // Follow up with tool results
        if (!empty($results)) {
            $followUp = "Based on these tool results:\n" . implode("\n", $results) . "\n\nPlease provide a final answer to: $originalQuestion";
            return $this->ask($followUp, array_merge($options, ['cache' => false]));
        }
        
        return $this->extractResponse($response);
    }
}

class LlamaAnthropicChat extends LlamaBaseChatProvider {
    protected function getDefaultModel(): string {
        return LlamaConfig::get('default_models.anthropic', 'claude-3-sonnet-20240229');
    }
    
    protected function getApiUrl(): string {
        return LlamaConfig::get('api_urls.anthropic') . '/messages';
    }
    
    protected function getHeaders(): array {
        $apiKey = LlamaConfig::get('api_keys.anthropic');
        if (!$apiKey) {
            throw new LlamaConfigException('Anthropic API key not configured');
        }
        
        return [
            'x-api-key: ' . $apiKey,
            'anthropic-version: 2023-06-01',
        ];
    }
    
    protected function buildRequest(string $question, array $options): array {
        $messages = array_merge($this->messages, [
            ['role' => 'user', 'content' => $question]
        ]);
        
        $request = [
            'model' => $this->model,
            'max_tokens' => $options['max_tokens'] ?? 2000,
            'messages' => $messages,
            'temperature' => $options['temperature'] ?? 0.7,
        ];
        
        if ($this->systemMessage) {
            $request['system'] = $this->systemMessage;
        }
        
        if (!empty($this->tools)) {
            $request['tools'] = array_map(fn($tool) => $tool->getSchema(), $this->tools);
        }
        
        if ($options['stream'] ?? false) {
            $request['stream'] = true;
        }
        
        return $request;
    }
    
    protected function extractResponse(array $response): string {
        if (isset($response['content'][0]['text'])) {
            return $response['content'][0]['text'];
        }
        return $response['content'][0]['text'] ?? '';
    }
    
    protected function extractStreamContent(array $data): ?string {
        if ($data['type'] === 'content_block_delta') {
            return $data['delta']['text'] ?? null;
        }
        return null;
    }
}

class LlamaOllamaChat extends LlamaBaseChatProvider {
    protected function getDefaultModel(): string {
        return LlamaConfig::get('default_models.ollama', 'llama3');
    }
    
    protected function getApiUrl(): string {
        return LlamaConfig::get('api_urls.ollama') . '/api/chat';
    }
    
    protected function getHeaders(): array {
        return []; // Ollama doesn't require authentication headers
    }
    
    protected function buildRequest(string $question, array $options): array {
        $messages = [];
        
        if ($this->systemMessage) {
            $messages[] = ['role' => 'system', 'content' => $this->systemMessage];
        }
        
        $messages = array_merge($messages, $this->messages);
        $messages[] = ['role' => 'user', 'content' => $question];
        
        return [
            'model' => $this->model,
            'messages' => $messages,
            'stream' => $options['stream'] ?? false,
            'options' => [
                'temperature' => $options['temperature'] ?? 0.7,
                'num_predict' => $options['max_tokens'] ?? 2000,
            ],
        ];
    }
    
    protected function extractResponse(array $response): string {
        return $response['message']['content'] ?? '';
    }
    
    protected function extractStreamContent(array $data): ?string {
        return $data['message']['content'] ?? null;
    }
}

// =============================================================================
// EMBEDDING IMPLEMENTATIONS WITH ADVANCED FEATURES
// =============================================================================

abstract class LlamaBaseEmbeddingProvider implements LlamaEmbeddingInterface {
    protected LlamaHttpClient $client;
    protected string $model;
    protected LlamaLoggerInterface $logger;
    protected LlamaCacheInterface $cache;
    
    public function __construct(string $model = null) {
        $this->model = $model ?? $this->getDefaultModel();
        $this->client = new LlamaHttpClient();
        $this->logger = LlamaLogger::getInstance();
        $this->cache = LlamaConfig::get('cache.enabled') ? new LlamaFileCache() : new LlamaMemoryCache();
    }
    
    abstract protected function getDefaultModel(): string;
    abstract protected function getApiUrl(): string;
    abstract protected function getHeaders(): array;
    abstract protected function buildRequest(array $texts): array;
    abstract protected function extractEmbeddings(array $response): array;
    abstract public function getDimension(): int;
    
    public function embed(string $text): array {
        return $this->embedBatch([$text])[0];
    }
    
    public function embedBatch(array $texts): array {
        if (empty($texts)) {
            return [];
        }
        
        return LlamaMetrics::timer('embedding.batch', function() use ($texts) {
            // Check cache first
            $cacheKeys = array_map(fn($text) => 'embed:' . md5($this->model . $text), $texts);
            $cached = [];
            $uncachedTexts = [];
            $uncachedIndices = [];
            
            foreach ($texts as $i => $text) {
                if ($this->cache->has($cacheKeys[$i])) {
                    $cached[$i] = $this->cache->get($cacheKeys[$i]);
                } else {
                    $uncachedTexts[] = $text;
                    $uncachedIndices[] = $i;
                }
            }
            
            $results = $cached;
            
            // Process uncached texts in batches
            if (!empty($uncachedTexts)) {
                $batchSize = LlamaConfig::get('batch_sizes.embedding', 100);
                $batches = array_chunk($uncachedTexts, $batchSize, true);
                
                foreach ($batches as $batch) {
                    $request = $this->buildRequest($batch);
                    $response = $this->client->post($this->getApiUrl(), $request, $this->getHeaders());
                    $embeddings = $this->extractEmbeddings($response);
                    
                    // Store in cache and results
                    $batchIndices = array_keys($batch);
                    foreach ($embeddings as $j => $embedding) {
                        $originalIndex = $uncachedIndices[$batchIndices[$j]];
                        $results[$originalIndex] = $embedding;
                        $this->cache->set($cacheKeys[$originalIndex], $embedding, 3600);
                    }
                }
            }
            
            // Sort by original indices
            ksort($results);
            
            LlamaMetrics::increment('embedding.requests', 1, ['model' => $this->model]);
            LlamaMetrics::gauge('embedding.batch_size', count($texts));
            
            return array_values($results);
        });
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

class LlamaOpenAIEmbeddings extends LlamaBaseEmbeddingProvider {
    private array $dimensions = [
        'text-embedding-3-small' => 1536,
        'text-embedding-3-large' => 3072,
        'text-embedding-ada-002' => 1536,
    ];
    
    protected function getDefaultModel(): string {
        return LlamaConfig::get('embedding_models.openai', 'text-embedding-3-large');
    }
    
    protected function getApiUrl(): string {
        return LlamaConfig::get('api_urls.openai') . '/embeddings';
    }
    
    protected function getHeaders(): array {
        $apiKey = LlamaConfig::get('api_keys.openai');
        if (!$apiKey) {
            throw new LlamaConfigException('OpenAI API key not configured');
        }
        
        return ['Authorization: Bearer ' . $apiKey];
    }
    
    protected function buildRequest(array $texts): array {
        return [
            'model' => $this->model,
            'input' => $texts,
            'encoding_format' => 'float',
        ];
    }
    
    protected function extractEmbeddings(array $response): array {
        return array_map(fn($item) => $item['embedding'], $response['data'] ?? []);
    }
    
    public function getDimension(): int {
        return $this->dimensions[$this->model] ?? 1536;
    }
}

class LlamaOllamaEmbeddings extends LlamaBaseEmbeddingProvider {
    protected function getDefaultModel(): string {
        return LlamaConfig::get('embedding_models.ollama', 'nomic-embed-text');
    }
    
    protected function getApiUrl(): string {
        return LlamaConfig::get('api_urls.ollama') . '/api/embeddings';
    }
    
    protected function getHeaders(): array {
        return [];
    }
    
    protected function buildRequest(array $texts): array {
        return [
            'model' => $this->model,
            'prompt' => $texts[0], // Ollama processes one at a time
        ];
    }
    
    protected function extractEmbeddings(array $response): array {
        return [$response['embedding'] ?? []];
    }
    
    public function embedBatch(array $texts): array {
        // Ollama doesn't support batch processing, so we process one by one
        $embeddings = [];
        foreach ($texts as $text) {
            $cacheKey = 'embed:' . md5($this->model . $text);
            
            if ($this->cache->has($cacheKey)) {
                $embeddings[] = $this->cache->get($cacheKey);
                continue;
            }
            
            $request = $this->buildRequest([$text]);
            $response = $this->client->post($this->getApiUrl(), $request, $this->getHeaders());
            $embedding = $this->extractEmbeddings($response)[0];
            
            $embeddings[] = $embedding;
            $this->cache->set($cacheKey, $embedding, 3600);
        }
        
        return $embeddings;
    }
    
    public function getDimension(): int {
        return 768; // Default for nomic-embed-text
    }
}

// =============================================================================
// ADVANCED VECTOR STORES
// =============================================================================

abstract class LlamaBaseVectorStore implements LlamaVectorStoreInterface {
    protected LlamaLoggerInterface $logger;
    
    public function __construct() {
        $this->logger = LlamaLogger::getInstance();
    }
    
    public function addBatch(array $documents): void {
        foreach ($documents as $doc) {
            if ($doc instanceof LlamaDocument) {
                $this->add($doc->id, $doc->embedding ?? [], $doc->content, $doc->metadata);
            }
        }
    }
    
    protected function cosineSimilarity(array $a, array $b): float {
        if (count($a) !== count($b) || empty($a)) {
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
    
    protected function euclideanDistance(array $a, array $b): float {
        if (count($a) !== count($b)) {
            return PHP_FLOAT_MAX;
        }
        
        $sum = 0.0;
        for ($i = 0; $i < count($a); $i++) {
            $diff = $a[$i] - $b[$i];
            $sum += $diff * $diff;
        }
        
        return sqrt($sum);
    }
}

class LlamaMemoryVectorStore extends LlamaBaseVectorStore {
    private array $vectors = [];
    private array $metadata = [];
    
    public function add(string $id, array $vector, string $content, array $metadata = []): void {
        $this->vectors[$id] = [
            'id' => $id,
            'vector' => $vector,
            'content' => $content,
            'metadata' => $metadata,
            'created_at' => time(),
        ];
        
        $this->metadata[$id] = $metadata;
        $this->logger->debug("Added vector to memory store", ['id' => $id, 'dimension' => count($vector)]);
    }
    
    public function search(array $vector, int $limit = 5, array $filters = []): array {
        $similarities = [];
        
        foreach ($this->vectors as $item) {
            // Apply filters
            if (!$this->matchesFilters($item['metadata'], $filters)) {
                continue;
            }
            
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
    
    public function delete(string $id): bool {
        if (isset($this->vectors[$id])) {
            unset($this->vectors[$id], $this->metadata[$id]);
            return true;
        }
        return false;
    }
    
    public function update(string $id, array $data): bool {
        if (!isset($this->vectors[$id])) {
            return false;
        }
        
        $this->vectors[$id] = array_merge($this->vectors[$id], $data);
        return true;
    }
    
    public function count(): int {
        return count($this->vectors);
    }
    
    public function clear(): void {
        $this->vectors = [];
        $this->metadata = [];
    }
    
    public function createIndex(array $options = []): bool {
        // Memory store doesn't need indexing
        return true;
    }
    
    private function matchesFilters(array $metadata, array $filters): bool {
        foreach ($filters as $key => $value) {
            if (!isset($metadata[$key]) || $metadata[$key] !== $value) {
                return false;
            }
        }
        return true;
    }
}

class LlamaSQLiteVectorStore extends LlamaBaseVectorStore {
    private \PDO $pdo;
    private string $tableName;
    
    public function __construct(string $dbPath = ':memory:', string $tableName = 'vectors') {
        parent::__construct();
        $this->tableName = $tableName;
        
        $this->pdo = new \PDO("sqlite:$dbPath");
        $this->pdo->setAttribute(\PDO::ATTR_ERRMODE, \PDO::ERRMODE_EXCEPTION);
        $this->createTables();
    }
    
    private function createTables(): void {
        $sql = "
            CREATE TABLE IF NOT EXISTS {$this->tableName} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                vector TEXT NOT NULL,
                metadata TEXT,
                created_at INTEGER DEFAULT (unixepoch()),
                updated_at INTEGER DEFAULT (unixepoch())
            );
            
            CREATE INDEX IF NOT EXISTS idx_{$this->tableName}_created_at ON {$this->tableName} (created_at);
        ";
        
        $this->pdo->exec($sql);
    }
    
    public function add(string $id, array $vector, string $content, array $metadata = []): void {
        $sql = "INSERT OR REPLACE INTO {$this->tableName} (id, content, vector, metadata, updated_at) 
                VALUES (?, ?, ?, ?, unixepoch())";
        
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute([
            $id,
            $content,
            json_encode($vector),
            json_encode($metadata),
        ]);
        
        $this->logger->debug("Added vector to SQLite store", ['id' => $id]);
    }
    
    public function search(array $vector, int $limit = 5, array $filters = []): array {
        // Basic vector search - in production, you'd use a proper vector extension
        $sql = "SELECT * FROM {$this->tableName}";
        $params = [];
        
        if (!empty($filters)) {
            $conditions = [];
            foreach ($filters as $key => $value) {
                $conditions[] = "json_extract(metadata, '$.$key') = ?";
                $params[] = $value;
            }
            $sql .= " WHERE " . implode(" AND ", $conditions);
        }
        
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute($params);
        
        $results = [];
        while ($row = $stmt->fetch(\PDO::FETCH_ASSOC)) {
            $storedVector = json_decode($row['vector'], true);
            $similarity = $this->cosineSimilarity($vector, $storedVector);
            
            $results[] = [
                'id' => $row['id'],
                'content' => $row['content'],
                'vector' => $storedVector,
                'metadata' => json_decode($row['metadata'], true),
                'similarity' => $similarity,
                'created_at' => $row['created_at'],
            ];
        }
        
        // Sort by similarity and limit
        usort($results, fn($a, $b) => $b['similarity'] <=> $a['similarity']);
        return array_slice($results, 0, $limit);
    }
    
    public function get(string $id): ?array {
        $sql = "SELECT * FROM {$this->tableName} WHERE id = ?";
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute([$id]);
        
        $row = $stmt->fetch(\PDO::FETCH_ASSOC);
        if (!$row) {
            return null;
        }
        
        return [
            'id' => $row['id'],
            'content' => $row['content'],
            'vector' => json_decode($row['vector'], true),
            'metadata' => json_decode($row['metadata'], true),
            'created_at' => $row['created_at'],
        ];
    }
    
    public function delete(string $id): bool {
        $sql = "DELETE FROM {$this->tableName} WHERE id = ?";
        $stmt = $this->pdo->prepare($sql);
        return $stmt->execute([$id]);
    }
    
    public function update(string $id, array $data): bool {
        $setClauses = [];
        $params = [];
        
        foreach ($data as $key => $value) {
            if (in_array($key, ['content', 'vector', 'metadata'])) {
                $setClauses[] = "$key = ?";
                $params[] = is_array($value) ? json_encode($value) : $value;
            }
        }
        
        if (empty($setClauses)) {
            return false;
        }
        
        $setClauses[] = "updated_at = unixepoch()";
        $params[] = $id;
        
        $sql = "UPDATE {$this->tableName} SET " . implode(", ", $setClauses) . " WHERE id = ?";
        $stmt = $this->pdo->prepare($sql);
        return $stmt->execute($params);
    }
    
    public function count(): int {
        $sql = "SELECT COUNT(*) FROM {$this->tableName}";
        return (int)$this->pdo->query($sql)->fetchColumn();
    }
    
    public function clear(): void {
        $this->pdo->exec("DELETE FROM {$this->tableName}");
    }
    
    public function createIndex(array $options = []): bool {
        // Create additional indexes based on options
        if ($options['metadata_keys'] ?? false) {
            foreach ($options['metadata_keys'] as $key) {
                $indexName = "idx_{$this->tableName}_meta_$key";
                $sql = "CREATE INDEX IF NOT EXISTS $indexName ON {$this->tableName} (json_extract(metadata, '$.$key'))";
                $this->pdo->exec($sql);
            }
        }
        
        return true;
    }
}

class LlamaPostgreSQLVectorStore extends LlamaBaseVectorStore {
    private \PDO $pdo;
    private string $tableName;
    
    public function __construct(array $config = [], string $tableName = 'vectors') {
        parent::__construct();
        $this->tableName = $tableName;
        
        $config = array_merge(LlamaConfig::get('vector_store.connection', []), $config);
        
        $dsn = "pgsql:host={$config['host']};port={$config['port']};dbname={$config['database']}";
        $this->pdo = new \PDO($dsn, $config['username'], $config['password'], [
            \PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION,
        ]);
        
        $this->createTables();
    }
    
    private function createTables(): void {
        // Create vector extension if not exists
        $this->pdo->exec("CREATE EXTENSION IF NOT EXISTS vector;");
        
        $sql = "
            CREATE TABLE IF NOT EXISTS {$this->tableName} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                vector VECTOR(1536),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS {$this->tableName}_vector_idx ON {$this->tableName} 
            USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
            
            CREATE INDEX IF NOT EXISTS {$this->tableName}_metadata_idx ON {$this->tableName} USING GIN (metadata);
        ";
        
        $this->pdo->exec($sql);
    }
    
    public function add(string $id, array $vector, string $content, array $metadata = []): void {
        $sql = "INSERT INTO {$this->tableName} (id, content, vector, metadata, updated_at) 
                VALUES (?, ?, ?, ?, NOW())
                ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                vector = EXCLUDED.vector,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()";
        
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute([
            $id,
            $content,
            '[' . implode(',', $vector) . ']',
            json_encode($metadata),
        ]);
        
        $this->logger->debug("Added vector to PostgreSQL store", ['id' => $id]);
    }
    
    public function search(array $vector, int $limit = 5, array $filters = []): array {
        $sql = "SELECT id, content, vector, metadata, 
                1 - (vector <=> ?) AS similarity
                FROM {$this->tableName}";
        
        $params = ['[' . implode(',', $vector) . ']'];
        
        if (!empty($filters)) {
            $conditions = [];
            foreach ($filters as $key => $value) {
                $conditions[] = "metadata->>? = ?";
                $params[] = $key;
                $params[] = $value;
            }
            $sql .= " WHERE " . implode(" AND ", $conditions);
        }
        
        $sql .= " ORDER BY similarity DESC LIMIT ?";
        $params[] = $limit;
        
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute($params);
        
        $results = [];
        while ($row = $stmt->fetch(\PDO::FETCH_ASSOC)) {
            // Parse vector string back to array
            $vectorStr = trim($row['vector'], '[]');
            $vectorArray = array_map('floatval', explode(',', $vectorStr));
            
            $results[] = [
                'id' => $row['id'],
                'content' => $row['content'],
                'vector' => $vectorArray,
                'metadata' => json_decode($row['metadata'], true),
                'similarity' => (float)$row['similarity'],
            ];
        }
        
        return $results;
    }
    
    public function get(string $id): ?array {
        $sql = "SELECT * FROM {$this->tableName} WHERE id = ?";
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute([$id]);
        
        $row = $stmt->fetch(\PDO::FETCH_ASSOC);
        if (!$row) {
            return null;
        }
        
        $vectorStr = trim($row['vector'], '[]');
        $vectorArray = array_map('floatval', explode(',', $vectorStr));
        
        return [
            'id' => $row['id'],
            'content' => $row['content'],
            'vector' => $vectorArray,
            'metadata' => json_decode($row['metadata'], true),
            'created_at' => $row['created_at'],
        ];
    }
    
    public function delete(string $id): bool {
        $sql = "DELETE FROM {$this->tableName} WHERE id = ?";
        $stmt = $this->pdo->prepare($sql);
        return $stmt->execute([$id]);
    }
    
    public function update(string $id, array $data): bool {
        $setClauses = [];
        $params = [];
        
        if (isset($data['content'])) {
            $setClauses[] = "content = ?";
            $params[] = $data['content'];
        }
        
        if (isset($data['vector'])) {
            $setClauses[] = "vector = ?";
            $params[] = '[' . implode(',', $data['vector']) . ']';
        }
        
        if (isset($data['metadata'])) {
            $setClauses[] = "metadata = ?";
            $params[] = json_encode($data['metadata']);
        }
        
        if (empty($setClauses)) {
            return false;
        }
        
        $setClauses[] = "updated_at = NOW()";
        $params[] = $id;
        
        $sql = "UPDATE {$this->tableName} SET " . implode(", ", $setClauses) . " WHERE id = ?";
        $stmt = $this->pdo->prepare($sql);
        return $stmt->execute($params);
    }
    
    public function count(): int {
        $sql = "SELECT COUNT(*) FROM {$this->tableName}";
        return (int)$this->pdo->query($sql)->fetchColumn();
    }
    
    public function clear(): void {
        $this->pdo->exec("TRUNCATE TABLE {$this->tableName}");
    }
    
    public function createIndex(array $options = []): bool {
        if ($options['recreate_vector_index'] ?? false) {
            $this->pdo->exec("DROP INDEX IF EXISTS {$this->tableName}_vector_idx");
            $lists = $options['ivf_lists'] ?? 100;
            $this->pdo->exec("
                CREATE INDEX {$this->tableName}_vector_idx ON {$this->tableName} 
                USING ivfflat (vector vector_cosine_ops) WITH (lists = $lists)
            ");
        }
        
        return true;
    }
}

// =============================================================================
// ADVANCED RAG SYSTEM
// =============================================================================

interface LlamaDocumentTransformerInterface {
    public function transform(array $documents, string $query): array;
}

class LlamaReranker implements LlamaDocumentTransformerInterface {
    private LlamaChatInterface $chat;
    private int $topK;
    
    public function __construct(LlamaChatInterface $chat, int $topK = 5) {
        $this->chat = $chat;
        $this->topK = $topK;
    }
    
    public function transform(array $documents, string $query): array {
        if (count($documents) <= $this->topK) {
            return $documents;
        }
        
        // Create reranking prompt
        $docTexts = array_map(fn($doc, $i) => "$i. " . $doc['content'], $documents, array_keys($documents));
        $prompt = "Given the query: \"$query\"\n\n";
        $prompt .= "Rank the following documents by relevance (most relevant first):\n\n";
        $prompt .= implode("\n\n", $docTexts);
        $prompt .= "\n\nRespond with only the document numbers in order of relevance (e.g., 3,1,5,2,4):";
        
        try {
            $response = $this->chat->ask($prompt);
            $order = array_map('intval', array_map('trim', explode(',', $response)));
            
            $reranked = [];
            foreach ($order as $index) {
                if (isset($documents[$index])) {
                    $reranked[] = $documents[$index];
                    if (count($reranked) >= $this->topK) {
                        break;
                    }
                }
            }
            
            return array_slice($reranked, 0, $this->topK);
        } catch (Exception $e) {
            // Fallback to original order
            return array_slice($documents, 0, $this->topK);
        }
    }
}

class LlamaMultiQueryTransformer implements LlamaDocumentTransformerInterface {
    private LlamaChatInterface $chat;
    private LlamaEmbeddingInterface $embedder;
    private int $numQueries;
    
    public function __construct(LlamaChatInterface $chat, LlamaEmbeddingInterface $embedder, int $numQueries = 3) {
        $this->chat = $chat;
        $this->embedder = $embedder;
        $this->numQueries = $numQueries;
    }
    
    public function transform(array $documents, string $query): array {
        // Generate alternative queries
        $prompt = "Given this question: \"$query\"\n\n";
        $prompt .= "Generate {$this->numQueries} different but related questions that would help find the same information. ";
        $prompt .= "Respond with one question per line:";
        
        try {
            $response = $this->chat->ask($prompt);
            $queries = array_filter(array_map('trim', explode("\n", $response)));
            $queries[] = $query; // Include original query
            
            // This would need the vector store to search with multiple queries
            // For now, just return the original documents
            return $documents;
        } catch (Exception $e) {
            return $documents;
        }
    }
}

class LlamaRAG {
    private LlamaVectorStoreInterface $vectorStore;
    private LlamaEmbeddingInterface $embedder;
    private LlamaChatInterface $chat;
    private LlamaDocumentProcessor $processor;
    private ?LlamaDocumentTransformerInterface $documentTransformer = null;
    private array $systemPrompts;
    private LlamaLoggerInterface $logger;
    private LlamaMetrics $metrics;
    
    public function __construct(
        string $chatProvider = 'openai',
        string $embeddingProvider = 'openai',
        string $vectorStoreType = 'memory',
        array $vectorStoreConfig = []
    ) {
        $this->logger = LlamaLogger::getInstance();
        $this->processor = new LlamaDocumentProcessor();
        
        // Initialize components
        $this->initializeVectorStore($vectorStoreType, $vectorStoreConfig);
        $this->initializeEmbedder($embeddingProvider);
        $this->initializeChat($chatProvider);
        
        $this->systemPrompts = [
            'default' => "You are a helpful AI assistant. Use the provided context to answer questions accurately. If the context doesn't contain enough information, say so clearly.",
            'conversational' => "You are a friendly and knowledgeable assistant. Answer questions based on the context provided, and feel free to ask clarifying questions if needed.",
            'analytical' => "You are an analytical AI that provides detailed, well-reasoned answers based on the available context. Include relevant details and explain your reasoning.",
        ];
    }
    
    private function initializeVectorStore(string $type, array $config): void {
        $this->vectorStore = match($type) {
            'memory' => new LlamaMemoryVectorStore(),
            'sqlite' => new LlamaSQLiteVectorStore($config['path'] ?? ':memory:'),
            'postgresql' => new LlamaPostgreSQLVectorStore($config),
            default => throw new LlamaException("Unsupported vector store type: $type"),
        };
    }
    
    private function initializeEmbedder(string $provider): void {
        $embeddings = new LlamaEmbeddings($provider);
        $this->embedder = match($provider) {
            'openai' => new LlamaOpenAIEmbeddings(),
            'ollama' => new LlamaOllamaEmbeddings(),
            default => throw new LlamaException("Unsupported embedding provider: $provider"),
        };
    }
    
    private function initializeChat(string $provider): void {
        $this->chat = match($provider) {
            'openai' => new LlamaOpenAIChat(),
            'anthropic' => new LlamaAnthropicChat(),
            'ollama' => new LlamaOllamaChat(),
            default => throw new LlamaException("Unsupported chat provider: $provider"),
        };
        
        $this->chat->setSystemMessage($this->systemPrompts['default']);
    }
    
    public function setDocumentTransformer(LlamaDocumentTransformerInterface $transformer): void {
        $this->documentTransformer = $transformer;
    }
    
    public function setSystemPromptStyle(string $style): void {
        if (isset($this->systemPrompts[$style])) {
            $this->chat->setSystemMessage($this->systemPrompts[$style]);
        }
    }
    
    public function addDocument(string $content, string $id = null, array $metadata = []): void {
        $doc = new LlamaDocument($content, $id, $metadata);
        $embedding = $this->embedder->embed($content);
        $this->vectorStore->add($doc->id, $embedding, $content, $metadata);
        
        $this->logger->info("Added document to RAG", ['id' => $doc->id, 'content_length' => strlen($content)]);
    }
    
    public function addDocuments(array $documents): void {
        if (empty($documents)) {
            return;
        }
        
        // Batch process embeddings
        $contents = [];
        $processedDocs = [];
        
        foreach ($documents as $doc) {
            if (is_string($doc)) {
                $document = new LlamaDocument($doc);
                $contents[] = $document->content;
                $processedDocs[] = $document;
            } elseif ($doc instanceof LlamaDocument) {
                $contents[] = $doc->content;
                $processedDocs[] = $doc;
            }
        }
        
        $embeddings = $this->embedder->embedBatch($contents);
        
        foreach ($processedDocs as $i => $doc) {
            $this->vectorStore->add($doc->id, $embeddings[$i], $doc->content, $doc->metadata);
        }
        
        $this->logger->info("Added batch of documents to RAG", ['count' => count($processedDocs)]);
    }
    
    public function loadText(string $text, int $chunkSize = 1000, string $strategy = 'tokens'): void {
        $chunks = LlamaDocumentSplitter::split($text, $chunkSize, 200, $strategy);
        $this->addDocuments($chunks);
    }
    
    public function loadFile(string $filename, int $chunkSize = 1000, string $strategy = 'tokens'): void {
        $documents = $this->processor->loadFile($filename);
        
        foreach ($documents as $doc) {
            $chunks = LlamaDocumentSplitter::split($doc->content, $chunkSize, 200, $strategy);
            foreach ($chunks as $chunk) {
                $chunk->metadata = array_merge($doc->metadata, $chunk->metadata);
            }
            $this->addDocuments($chunks);
        }
    }
    
    public function loadDirectory(string $directory, int $chunkSize = 1000, bool $recursive = true): void {
        $documents = $this->processor->loadDirectory($directory, $recursive);
        
        foreach ($documents as $doc) {
            $chunks = LlamaDocumentSplitter::split($doc->content, $chunkSize, 200);
            foreach ($chunks as $chunk) {
                $chunk->metadata = array_merge($doc->metadata, $chunk->metadata);
            }
            $this->addDocuments($chunks);
        }
    }
    
    public function loadUrl(string $url, int $chunkSize = 1000): void {
        $documents = $this->processor->loadUrl($url);
        
        foreach ($documents as $doc) {
            $chunks = LlamaDocumentSplitter::split($doc->content, $chunkSize, 200);
            foreach ($chunks as $chunk) {
                $chunk->metadata = array_merge($doc->metadata, $chunk->metadata);
            }
            $this->addDocuments($chunks);
        }
    }
    
    public function ask(string $question, array $options = []): string {
        return LlamaMetrics::timer('rag.ask', function() use ($question, $options) {
            $contextLimit = $options['context_limit'] ?? 5;
            $includeMetadata = $options['include_metadata'] ?? false;
            $filters = $options['filters'] ?? [];
            
            // Get question embedding
            $questionEmbedding = $this->embedder->embed($question);
            
            // Search for relevant documents
            $results = $this->vectorStore->search($questionEmbedding, $contextLimit * 2, $filters);
            
            if (empty($results)) {
                $response = $this->chat->ask($question, $options);
                $this->logger->warning("No context found for question", ['question' => $question]);
                return $response;
            }
            
            // Apply document transformer if configured
            if ($this->documentTransformer) {
                $results = $this->documentTransformer->transform($results, $question);
            }
            
            $results = array_slice($results, 0, $contextLimit);
            
            // Build context from search results
            $context = $this->buildContext($results, $includeMetadata);
            
            // Create enhanced prompt
            $enhancedPrompt = $this->buildPrompt($context, $question, $options);
            
            // Get response from chat
            $response = $this->chat->ask($enhancedPrompt, array_merge($options, ['cache' => false]));
            
            // Log metrics
            LlamaMetrics::increment('rag.questions', 1);
            LlamaMetrics::gauge('rag.context_docs', count($results));
            LlamaMetrics::gauge('rag.context_length', strlen($context));
            
            return $response;
        });
    }
    
    public function askWithSources(string $question, array $options = []): array {
        $contextLimit = $options['context_limit'] ?? 5;
        
        // Get relevant documents
        $questionEmbedding = $this->embedder->embed($question);
        $results = $this->vectorStore->search($questionEmbedding, $contextLimit, $options['filters'] ?? []);
        
        if ($this->documentTransformer) {
            $results = $this->documentTransformer->transform($results, $question);
        }
        
        $results = array_slice($results, 0, $contextLimit);
        
        // Get answer
        $answer = $this->ask($question, $options);
        
        // Format sources
        $sources = array_map(function($result, $index) {
            return [
                'index' => $index + 1,
                'content' => substr($result['content'], 0, 200) . '...',
                'similarity' => round($result['similarity'], 3),
                'metadata' => $result['metadata'] ?? [],
            ];
        }, $results, array_keys($results));
        
        return [
            'answer' => $answer,
            'sources' => $sources,
            'total_sources' => count($results),
        ];
    }
    
    public function getRelevantDocuments(string $question, int $limit = 5, array $filters = []): array {
        $questionEmbedding = $this->embedder->embed($question);
        return $this->vectorStore->search($questionEmbedding, $limit, $filters);
    }
    
    public function getStats(): array {
        return [
            'total_documents' => $this->vectorStore->count(),
            'embedding_dimension' => $this->embedder->getDimension(),
            'chat_usage' => $this->chat->getUsage(),
        ];
    }
    
    private function buildContext(array $results, bool $includeMetadata = false): string {
        $context = "Context information:\n\n";
        
        foreach ($results as $i => $result) {
            $context .= "Document " . ($i + 1) . ":\n";
            $context .= $result['content'] . "\n";
            
            if ($includeMetadata && !empty($result['metadata'])) {
                $metadataStr = json_encode($result['metadata'], JSON_PRETTY_PRINT);
                $context .= "Metadata: $metadataStr\n";
            }
            
            $context .= "\n";
        }
        
        return $context;
    }
    
    private function buildPrompt(string $context, string $question, array $options): string {
        $promptTemplate = $options['prompt_template'] ?? null;
        
        if ($promptTemplate) {
            return str_replace(['{context}', '{question}'], [$context, $question], $promptTemplate);
        }
        
        $prompt = $context . "\n";
        $prompt .= "Question: $question\n\n";
        $prompt .= "Please answer the question based on the context information provided above. ";
        $prompt .= "If the context doesn't contain enough information to answer the question, ";
        $prompt .= "please say so clearly and explain what information would be needed.";
        
        return $prompt;
    }
}

// =============================================================================
// EVALUATION FRAMEWORK
// =============================================================================

interface LlamaEvaluatorInterface {
    public function evaluate(string $generated, string $reference = null, array $context = []): array;
}

class LlamaStringComparisonEvaluator implements LlamaEvaluatorInterface {
    public function evaluate(string $generated, string $reference = null, array $context = []): array {
        if ($reference === null) {
            throw new LlamaValidationException("Reference text required for string comparison evaluation");
        }
        
        return [
            'rouge_1' => $this->calculateRouge1($generated, $reference),
            'rouge_2' => $this->calculateRouge2($generated, $reference),
            'rouge_l' => $this->calculateRougeL($generated, $reference),
            'bleu' => $this->calculateBleu($generated, $reference),
            'exact_match' => $generated === $reference ? 1.0 : 0.0,
            'length_ratio' => strlen($generated) / strlen($reference),
        ];
    }
    
    private function calculateRouge1(string $generated, string $reference): float {
        $genWords = $this->tokenize($generated);
        $refWords = $this->tokenize($reference);
        
        if (empty($refWords)) return 0.0;
        
        $overlap = count(array_intersect($genWords, $refWords));
        return $overlap / count($refWords);
    }
    
    private function calculateRouge2(string $generated, string $reference): float {
        $genBigrams = $this->getBigrams($generated);
        $refBigrams = $this->getBigrams($reference);
        
        if (empty($refBigrams)) return 0.0;
        
        $overlap = count(array_intersect($genBigrams, $refBigrams));
        return $overlap / count($refBigrams);
    }
    
    private function calculateRougeL(string $generated, string $reference): float {
        $genWords = $this->tokenize($generated);
        $refWords = $this->tokenize($reference);
        
        $lcs = $this->longestCommonSubsequence($genWords, $refWords);
        
        if (empty($refWords)) return 0.0;
        
        return $lcs / count($refWords);
    }
    
    private function calculateBleu(string $generated, string $reference): float {
        // Simplified BLEU-1 score
        $genWords = $this->tokenize($generated);
        $refWords = $this->tokenize($reference);
        
        if (empty($genWords) || empty($refWords)) return 0.0;
        
        $overlap = count(array_intersect($genWords, $refWords));
        $precision = $overlap / count($genWords);
        
        // Brevity penalty
        $bp = count($genWords) < count($refWords) ? 
              exp(1 - count($refWords) / count($genWords)) : 1.0;
        
        return $bp * $precision;
    }
    
    private function tokenize(string $text): array {
        return array_filter(preg_split('/\s+/', strtolower(trim($text))));
    }
    
    private function getBigrams(string $text): array {
        $words = $this->tokenize($text);
        $bigrams = [];
        
        for ($i = 0; $i < count($words) - 1; $i++) {
            $bigrams[] = $words[$i] . ' ' . $words[$i + 1];
        }
        
        return $bigrams;
    }
    
    private function longestCommonSubsequence(array $a, array $b): int {
        $m = count($a);
        $n = count($b);
        $dp = array_fill(0, $m + 1, array_fill(0, $n + 1, 0));
        
        for ($i = 1; $i <= $m; $i++) {
            for ($j = 1; $j <= $n; $j++) {
                if ($a[$i - 1] === $b[$j - 1]) {
                    $dp[$i][$j] = $dp[$i - 1][$j - 1] + 1;
                } else {
                    $dp[$i][$j] = max($dp[$i - 1][$j], $dp[$i][$j - 1]);
                }
            }
        }
        
        return $dp[$m][$n];
    }
}

class LlamaSemanticEvaluator implements LlamaEvaluatorInterface {
    private LlamaEmbeddingInterface $embedder;
    
    public function __construct(LlamaEmbeddingInterface $embedder) {
        $this->embedder = $embedder;
    }
    
    public function evaluate(string $generated, string $reference = null, array $context = []): array {
        if ($reference === null) {
            throw new LlamaValidationException("Reference text required for semantic evaluation");
        }
        
        $genEmbedding = $this->embedder->embed($generated);
        $refEmbedding = $this->embedder->embed($reference);
        
        $similarity = $this->cosineSimilarity($genEmbedding, $refEmbedding);
        
        return [
            'semantic_similarity' => $similarity,
            'semantic_distance' => 1.0 - $similarity,
        ];
    }
    
    private function cosineSimilarity(array $a, array $b): float {
        if (count($a) !== count($b) || empty($a)) {
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
}

// =============================================================================
// MAIN FACADE CLASSES
// =============================================================================

class LlamaChat {
    private LlamaChatInterface $chat;
    private string $provider;
    
    public function __construct(string $provider = 'openai', string $model = null) {
        $this->provider = $provider;
        LlamaConfig::load();
        
        $this->chat = match(strtolower($provider)) {
            'openai' => new LlamaOpenAIChat($model),
            'anthropic' => new LlamaAnthropicChat($model),
            'ollama' => new LlamaOllamaChat($model),
            default => throw new LlamaException("Unsupported provider: $provider"),
        };
    }
    
    public function ask(string $question, array $options = []): string {
        return $this->chat->ask($question, $options);
    }
    
    public function askStream(string $question, callable $callback = null): Generator {
        return $this->chat->askStream($question, $callback);
    }
    
    public function setSystemMessage(string $message): void {
        $this->chat->setSystemMessage($message);
    }
    
    public function addTool(LlamaFunctionTool $tool): void {
        $this->chat->addTool($tool);
    }
    
    public function getUsage(): array {
        return $this->chat->getUsage();
    }
    
    public function getProvider(): string {
        return $this->provider;
    }
}

class LlamaEmbeddings {
    private LlamaEmbeddingInterface $embedder;
    private string $provider;
    
    public function __construct(string $provider = 'openai', string $model = null) {
        $this->provider = $provider;
        LlamaConfig::load();
        
        $this->embedder = match(strtolower($provider)) {
            'openai' => new LlamaOpenAIEmbeddings($model),
            'ollama' => new LlamaOllamaEmbeddings($model),
            default => throw new LlamaException("Unsupported embedding provider: $provider"),
        };
    }
    
    public function embed(string $text): array {
        return $this->embedder->embed($text);
    }
    
    public function embedBatch(array $texts): array {
        return $this->embedder->embedBatch($texts);
    }
    
    public function getDimension(): int {
        return $this->embedder->getDimension();
    }
    
    public function getProvider(): string {
        return $this->provider;
    }
}

// =============================================================================
// HEALTH CHECK & MONITORING
// =============================================================================

class LlamaHealthCheck {
    public static function checkSystem(): array {
        $checks = [
            'php_version' => self::checkPhpVersion(),
            'extensions' => self::checkExtensions(),
            'config' => self::checkConfiguration(),
            'connectivity' => self::checkConnectivity(),
            'storage' => self::checkStorage(),
        ];
        
        $overall = array_reduce($checks, fn($carry, $check) => $carry && $check['status'] === 'ok', true);
        
        return [
            'status' => $overall ? 'ok' : 'error',
            'checks' => $checks,
            'timestamp' => date('c'),
        ];
    }
    
    private static function checkPhpVersion(): array {
        $required = '8.1.0';
        $current = PHP_VERSION;
        $ok = version_compare($current, $required, '>=');
        
        return [
            'status' => $ok ? 'ok' : 'error',
            'message' => "PHP $current (required: $required+)",
            'details' => ['current' => $current, 'required' => $required],
        ];
    }
    
    private static function checkExtensions(): array {
        $required = ['curl', 'json', 'pdo'];
        $optional = ['zip', 'gd', 'pgsql', 'sqlite3'];
        
        $missing = array_filter($required, fn($ext) => !extension_loaded($ext));
        $available = array_filter($optional, fn($ext) => extension_loaded($ext));
        
        return [
            'status' => empty($missing) ? 'ok' : 'error',
            'message' => empty($missing) ? 'All required extensions available' : 'Missing extensions: ' . implode(', ', $missing),
            'details' => [
                'required' => $required,
                'missing' => $missing,
                'optional_available' => $available,
            ],
        ];
    }
    
    private static function checkConfiguration(): array {
        $issues = [];
        
        // Check API keys
        $providers = ['openai', 'anthropic'];
        foreach ($providers as $provider) {
            $apiKey = LlamaConfig::get("api_keys.$provider");
            if (!$apiKey) {
                $issues[] = "No API key configured for $provider";
            }
        }
        
        // Check timeouts
        $timeout = LlamaConfig::get('timeouts.default');
        if ($timeout < 5 || $timeout > 300) {
            $issues[] = "Default timeout should be between 5-300 seconds";
        }
        
        return [
            'status' => empty($issues) ? 'ok' : 'warning',
            'message' => empty($issues) ? 'Configuration looks good' : 'Configuration issues found',
            'details' => $issues,
        ];
    }
    
    private static function checkConnectivity(): array {
        $endpoints = [
            'openai' => 'https://api.openai.com/v1/models',
            'anthropic' => 'https://api.anthropic.com/v1/complete',
        ];
        
        $results = [];
        foreach ($endpoints as $provider => $url) {
            $apiKey = LlamaConfig::get("api_keys.$provider");
            if (!$apiKey) {
                $results[$provider] = ['status' => 'skipped', 'reason' => 'No API key'];
                continue;
            }
            
            try {
                $client = new LlamaHttpClient(['timeout' => 5]);
                $headers = match($provider) {
                    'openai' => ["Authorization: Bearer $apiKey"],
                    'anthropic' => ["x-api-key: $apiKey", "anthropic-version: 2023-06-01"],
                    default => [],
                };
                
                $client->get($url, $headers);
                $results[$provider] = ['status' => 'ok'];
            } catch (Exception $e) {
                $results[$provider] = ['status' => 'error', 'error' => $e->getMessage()];
            }
        }
        
        $allOk = !array_filter($results, fn($r) => $r['status'] === 'error');
        
        return [
            'status' => $allOk ? 'ok' : 'error',
            'message' => $allOk ? 'All APIs accessible' : 'Some APIs not accessible',
            'details' => $results,
        ];
    }
    
    private static function checkStorage(): array {
        $issues = [];
        
        // Check cache directory
        $cacheDir = LlamaConfig::get('cache.path', sys_get_temp_dir() . '/llama_cache');
        if (!is_dir($cacheDir) && !mkdir($cacheDir, 0755, true)) {
            $issues[] = "Cannot create cache directory: $cacheDir";
        } elseif (!is_writable($cacheDir)) {
            $issues[] = "Cache directory not writable: $cacheDir";
        }
        
        // Check log directory
        $logFile = LlamaConfig::get('logging.file', '/tmp/llama.log');
        $logDir = dirname($logFile);
        if (!is_dir($logDir) && !mkdir($logDir, 0755, true)) {
            $issues[] = "Cannot create log directory: $logDir";
        } elseif (!is_writable($logDir)) {
            $issues[] = "Log directory not writable: $logDir";
        }
        
        return [
            'status' => empty($issues) ? 'ok' : 'error',
            'message' => empty($issues) ? 'Storage accessible' : 'Storage issues found',
            'details' => $issues,
        ];
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function llama_chat(string $question, string $provider = 'openai', array $options = []): string {
    static $instances = [];
    
    $key = "$provider:" . ($options['model'] ?? 'default');
    if (!isset($instances[$key])) {
        $instances[$key] = new LlamaChat($provider, $options['model'] ?? null);
        if (isset($options['system_message'])) {
            $instances[$key]->setSystemMessage($options['system_message']);
        }
    }
    
    return $instances[$key]->ask($question, $options);
}

function llama_embed(string $text, string $provider = 'openai'): array {
    static $instances = [];
    
    if (!isset($instances[$provider])) {
        $instances[$provider] = new LlamaEmbeddings($provider);
    }
    
    return $instances[$provider]->embed($text);
}

function llama_rag_ask(string $question, array $documents = [], array $options = []): string {
    static $rag = null;
    
    if ($rag === null) {
        $rag = new LlamaRAG(
            $options['chat_provider'] ?? 'openai',
            $options['embedding_provider'] ?? 'openai',
            $options['vector_store'] ?? 'memory'
        );
    }
    
    if (!empty($documents)) {
        $rag->addDocuments($documents);
    }
    
    return $rag->ask($question, $options);
}

function llama_health_check(): array {
    return LlamaHealthCheck::checkSystem();
}

function llama_metrics(): array {
    return LlamaMetrics::getMetrics();
}

function command_exists(string $command): bool {
    $result = shell_exec("which $command");
    return !empty($result);
}

// =============================================================================
// INITIALIZATION
// =============================================================================

// Auto-load configuration on include
LlamaConfig::load();

// Register error handler for better debugging
set_error_handler(function($severity, $message, $file, $line) {
    if (error_reporting() & $severity) {
        $logger = LlamaLogger::getInstance();
        $logger->error("PHP Error: $message", [
            'file' => $file,
            'line' => $line,
            'severity' => $severity,
        ]);
    }
    return false; // Don't prevent default error handling
});

// Register shutdown function for cleanup
register_shutdown_function(function() {
    $error = error_get_last();
    if ($error && in_array($error['type'], [E_ERROR, E_CORE_ERROR, E_COMPILE_ERROR, E_RECOVERABLE_ERROR])) {
        $logger = LlamaLogger::getInstance();
        $logger->critical("Fatal error: " . $error['message'], [
            'file' => $error['file'],
            'line' => $error['line'],
        ]);
    }
});

// =============================================================================
// EXAMPLE USAGE & DOCUMENTATION
// =============================================================================

/*
## ENTERPRISE USAGE EXAMPLES

### 1. Basic Chat with Multiple Providers
```php
// OpenAI
$chat = new LlamaChat('openai', 'gpt-4-turbo');
$chat->setSystemMessage("You are a helpful assistant.");
echo $chat->ask("What is machine learning?");

// Anthropic
$anthropic = new LlamaChat('anthropic', 'claude-3-sonnet-20240229');
echo $anthropic->ask("Explain quantum computing");

// Ollama (local)
$ollama = new LlamaChat('ollama', 'llama3');
echo $ollama->ask("What is PHP?");
```

### 2. Advanced RAG with PostgreSQL
```php
// Configure for production
LlamaConfig::set('vector_store.connection.host', 'db.example.com');
LlamaConfig::set('vector_store.connection.database', 'llama_prod');
LlamaConfig::set('vector_store.connection.username', 'llama_user');
LlamaConfig::setSecret('vector_store.connection.password', 'secure_password');

$rag = new LlamaRAG('openai', 'openai', 'postgresql');

// Load entire document collections
$rag->loadDirectory('/path/to/documentation', 1000, true);

// Add reranking for better results
$reranker = new LlamaReranker(new LlamaOpenAIChat(), 5);
$rag->setDocumentTransformer($reranker);

// Ask with metadata filtering
$answer = $rag->askWithSources("How do I configure the system?", [
    'filters' => ['document_type' => 'configuration'],
    'context_limit' => 10,
]);
```

### 3. Function Calling / Tools
```php
class WeatherService {
    public function getCurrentWeather(array $args): array {
        $location = $args['location'];
        // Call external API
        return ['temperature' => 22, 'condition' => 'sunny'];
    }
}

$weather = new WeatherService();
$weatherTool = new LlamaFunctionTool(
    'get_weather',
    'Get current weather for a location',
    [
        new LlamaFunctionParameter('location', 'string', 'City name', true)
    ],
    [$weather, 'getCurrentWeather']
);

$chat = new LlamaChat('openai');
$chat->addTool($weatherTool);
echo $chat->ask("What's the weather in Paris?");
```

### 4. Streaming Responses
```php
$chat = new LlamaChat('openai');
foreach ($chat->askStream("Write a long essay about AI") as $chunk) {
    echo $chunk;
    flush();
}
```

### 5. Evaluation Framework
```php
$evaluator = new LlamaStringComparisonEvaluator();
$results = $evaluator->evaluate(
    "Paris is the capital of France",
    "The capital of France is Paris"
);

$semanticEvaluator = new LlamaSemanticEvaluator(new LlamaOpenAIEmbeddings());
$similarity = $semanticEvaluator->evaluate("AI is powerful", "Artificial intelligence is strong");
```

### 6. Health Monitoring
```php
// Health check endpoint
header('Content-Type: application/json');
echo json_encode(llama_health_check());

// Metrics endpoint  
echo json_encode(llama_metrics());
```

### 7. Production Configuration
```php
// config/llama.php
LlamaConfig::load([
    'logging' => [
        'level' => 'WARNING',
        'file' => '/var/log/llama/app.log',
        'max_size' => '500MB',
    ],
    'cache' => [
        'enabled' => true,
        'type' => 'redis',
        'connection' => [
            'host' => 'redis.example.com',
            'port' => 6379,
        ],
    ],
    'rate_limits' => [
        'requests_per_minute' => 1000,
        'tokens_per_minute' => 1000000,
    ],
    'security' => [
        'enable_prompt_injection_detection' => true,
        'max_input_length' => 50000,
    ],
]);
```

### 8. Batch Processing
```php
$embedder = new LlamaEmbeddings('openai');
$texts = ["text1", "text2", "text3"]; // thousands of texts
$embeddings = $embedder->embedBatch($texts); // Efficient batch processing

$rag = new LlamaRAG();
$rag->addDocuments($documents); // Batch document addition
```

### 9. Document Processing Pipeline
```php
$processor = new LlamaDocumentProcessor();

// Load various file types
$docs = $processor->loadDirectory('/data/documents');

//