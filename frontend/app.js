// Generate or retrieve a persistent anonymous user ID
function getUserId() {
    let userId = localStorage.getItem('rag_user_id');
    if (!userId) {
        // Fallback
        if (typeof crypto !== 'undefined' && crypto.randomUUID) {
            userId = crypto.randomUUID();
        } else {
            userId = 'anon-' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
        }
        localStorage.setItem('rag_user_id', userId);
    }
    return userId;
}

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
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');

    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
    }

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
            formData.append('user_id', getUserId());

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

        const avatarIcon = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-cube"></i>';

        msgDiv.innerHTML = `
            <div class="avatar">${avatarIcon}</div>
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

        // Container for all sources
        const sourceSection = document.createElement('div');
        sourceSection.className = 'source-section';
        
        sourceSection.innerHTML = `
            <div class="source-header">Retrieved Sources</div>
            <div class="source-grid"></div>
        `;

        const grid = sourceSection.querySelector('.source-grid');

        sources.forEach((source) => {
            const card = document.createElement('div');
            card.className = 'source-card';

            const name = source.filename || 'Document';
            const scorePercent = Math.round(source.score * 100);

            card.innerHTML = `
                <div class="source-card-header" title="Click to view full snippet">
                    <div class="source-title-wrap">
                        <i class="fa-solid fa-file-alt"></i>
                        <span class="source-name">${name}</span>
                        <span class="source-score">${scorePercent}% MATCH</span>
                    </div>
                    <i class="fa-solid fa-chevron-down toggle-icon"></i>
                </div>
                <div class="source-snippet">${source.text || 'No snippet available.'}</div>
            `;

            // Toggle snippet functionality
            const header = card.querySelector('.source-card-header');
            const snippet = card.querySelector('.source-snippet');
            const icon = card.querySelector('.toggle-icon');
            
            header.addEventListener('click', () => {
                snippet.classList.toggle('active');
                icon.classList.toggle('fa-chevron-up');
                icon.classList.toggle('fa-chevron-down');
                // Ensure we stay scrolled to bottom if the div expands
                setTimeout(scrollToBottom, 50);
            });

            grid.appendChild(card);
        });

        // Prepend to the parent of message-content so it renders BEFORE the text response
        msgContentDiv.parentElement.insertBefore(sourceSection, msgContentDiv);
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
                body: JSON.stringify({ query, user_id: getUserId(), top_k: topK })
            });

            if (!response.ok) throw new Error('Query failed');

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');

                // Keep the last partial line in the buffer
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.trim().startsWith('data: ')) {
                        const dataStr = line.trim().substring(6).trim();
                        if (!dataStr) continue;
                        try {
                            const data = JSON.parse(dataStr);

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

                            // Handle sources 
                            if (data.sources) {
                                appendSources(assistantNodes.textContent, data.sources);
                                scrollToBottom();
                            }

                            // Handle completion containing latency
                            if (data.done) {
                                latencyMetric.textContent = `${data.latency_ms.toFixed(0)} ms`;
                            }
                        } catch (e) {
                            console.error("Failed to parse JSON:", dataStr, e);
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
