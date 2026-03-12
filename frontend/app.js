document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const topKSlider = document.getElementById('top-k-slider');
    const topKValue = document.getElementById('top-k-value');
    const chatForm = document.getElementById('chat-form');
    const queryInput = document.getElementById('query-input');
    const chatHistory = document.getElementById('chat-history');
    const sendButton = document.getElementById('send-button');
    const latencyMetric = document.getElementById('latency-metric');

    // API Base URL
    const API_BASE = 'http://18.219.12.24:8080';

    // File Upload Handling
    const handleDragEvent = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, handleDragEvent, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file) uploadFile(file);
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            uploadFile(e.target.files[0]);
        }
    });

    async function uploadFile(file) {
        if (file) {
            uploadStatus.textContent = `Uploading ${file.name}...`;
            uploadStatus.className = 'status-msg';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`${API_BASE}/documents/upload`, {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    uploadStatus.textContent = `Successfully uploaded ${file.name}`;
                    uploadStatus.className = 'status-msg status-success';
                } else {
                    throw new Error('Upload failed');
                }
            } catch (error) {
                uploadStatus.textContent = `Failed to upload: ${error.message}`;
                uploadStatus.className = 'status-msg status-error';
            }
        }
    }

    // Top K Slider
    topKSlider.addEventListener('input', (e) => {
        topKValue.textContent = e.target.value;
    });

    // Chat Interface
    function appendMessage(role, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role === 'user' ? 'user-msg' : 'system-msg'}`;

        const avatarIcon = role === 'user' ? 'fa-user' : 'fa-robot';

        msgDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid ${avatarIcon}"></i></div>
            <div class="message-content">
                <div class="text-content"></div>
            </div>
        `;

        const textContent = msgDiv.querySelector('.text-content');

        if (role === 'system-typing') {
            textContent.innerHTML = `
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
            // It will be replaced with streaming content, so return the inner container
            chatHistory.appendChild(msgDiv);
            scrollToBottom();
            return {
                msgContainer: msgDiv,
                textContent: textContent
            };
        } else {
            textContent.textContent = content; // initial text
            chatHistory.appendChild(msgDiv);
            scrollToBottom();
            return {
                msgContainer: msgDiv,
                textContent: textContent
            };
        }
    }

    function appendSources(msgContentDiv, sources) {
        if (!sources || sources.length === 0) return;

        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'source-badges';

        sources.forEach((source, index) => {
            const badge = document.createElement('div');
            badge.className = 'source-badge';

            // Extract filename if it's a path
            let filename = source.document_id;
            if (filename && filename.includes('/')) {
                filename = filename.substring(filename.lastIndexOf('/') + 1);
            } else if (filename && filename.includes('\\')) {
                filename = filename.substring(filename.lastIndexOf('\\') + 1);
            }

            badge.innerHTML = `<i class="fa-solid fa-file-lines"></i> ${filename || 'Document'} (Score: ${(source.score).toFixed(2)})`;
            sourcesDiv.appendChild(badge);
        });

        msgContentDiv.parentElement.appendChild(sourcesDiv);
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const query = queryInput.value.trim();
        if (!query) return;

        const topK = parseInt(topKSlider.value, 10);

        // 1. Add user message
        appendMessage('user', query);
        queryInput.value = '';
        queryInput.disabled = true;
        sendButton.disabled = true;

        // 2. Add empty assistant message with typing indicator
        const assistantNodes = appendMessage('system-typing', '');
        let accumulatedText = '';
        let hasStartedStreaming = false;

        try {
            const response = await fetch(`${API_BASE}/query/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: topK })
            });

            if (!response.ok) throw new Error('Query failed');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.substring(6));

                        // Handle streaming tokens
                        if (data.token) {
                            if (!hasStartedStreaming) {
                                assistantNodes.textContent.innerHTML = ''; // clear typing indicator
                                hasStartedStreaming = true;
                            }
                            accumulatedText += data.token;
                            assistantNodes.textContent.textContent = accumulatedText;
                            scrollToBottom();
                        }

                        // Handle completion containing latency and sources
                        if (data.done) {
                            latencyMetric.textContent = `${data.latency_ms.toFixed(0)} ms`;
                            if (data.sources) {
                                appendSources(assistantNodes.textContent, data.sources);
                                scrollToBottom();
                            }
                        }
                    }
                }
            }

            // Fallback clear typing indicator if no tokens arrived
            if (!hasStartedStreaming) {
                assistantNodes.textContent.textContent = 'No response generated.';
            }

        } catch (error) {
            assistantNodes.textContent.innerHTML = `<span style="color:var(--error)">Error generating response: ${error.message}</span>`;
        } finally {
            queryInput.disabled = false;
            sendButton.disabled = false;
            queryInput.focus();
        }
    });
});
