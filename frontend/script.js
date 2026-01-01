document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const spinner = document.getElementById('spinner');

    const API_URL = 'http://127.0.0.1:8000/analyze';
    const STORAGE_KEY = 'burnout_advisor_chats';

    // Load saved chats from localStorage
    const loadChats = () => {
        const savedChats = localStorage.getItem(STORAGE_KEY);
        if (savedChats) {
            const chats = JSON.parse(savedChats);
            chats.forEach((chat, index) => {
                const messageElement = document.createElement('div');
                messageElement.classList.add('message', `${chat.sender}-message`);
                const messageId = generateMessageId();
                messageElement.dataset.messageId = messageId;
                messageElement.dataset.pairIndex = Math.floor(index / 2); // Track conversation pairs
                messageElement.innerHTML = `
                    <div class="message-content">${chat.html}</div>
                    ${chat.sender === 'user' ? createMessageMenu(messageId) : ''}
                `;
                chatBox.appendChild(messageElement);
            });
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    };

    // Create the three-dot menu HTML
    const createMessageMenu = (messageId) => {
        return `
            <div class="message-menu">
                <button class="menu-trigger" onclick="toggleMenu('${messageId}')">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <circle cx="8" cy="3" r="1.5"/>
                        <circle cx="8" cy="8" r="1.5"/>
                        <circle cx="8" cy="13" r="1.5"/>
                    </svg>
                </button>
                <div class="menu-dropdown" id="menu-${messageId}">
                    <button class="menu-item" onclick="editMessage('${messageId}')">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                        Edit
                    </button>
                    <button class="menu-item menu-item-danger" onclick="deleteMessagePair('${messageId}')">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"/>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                        Delete
                    </button>
                </div>
            </div>
        `;
    };

    // Save chats to localStorage
    const saveChats = () => {
        const messages = chatBox.querySelectorAll('.message');
        const chats = [];
        messages.forEach(msg => {
            const sender = msg.classList.contains('user-message') ? 'user' : 'bot';
            const contentEl = msg.querySelector('.message-content');
            if (contentEl) {
                chats.push({ sender, html: contentEl.innerHTML });
            }
        });
        localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    };

    // Clear chat history
    const clearChats = () => {
        localStorage.removeItem(STORAGE_KEY);
        chatBox.innerHTML = '';
        // Add welcome message back
        const welcomeMsg = document.createElement('div');
        welcomeMsg.classList.add('message', 'bot-message');
        welcomeMsg.innerHTML = `<div class="message-content">🔮 Greetings, weary scholar! I am the Burnout Oracle, ancient keeper of academic wisdom! Shareth with me what troubles thy mind. How hast thou been feeling about thy scholarly pursuits? (Prithee, write at least 10 characters)</div>`;
        chatBox.appendChild(welcomeMsg);
    };

    // Expose clearChats globally so it can be called from the button
    window.clearChats = clearChats;

    // Generate unique ID for messages
    let messageIdCounter = Date.now();
    const generateMessageId = () => `msg-${messageIdCounter++}`;

    const addMessage = (text, sender, save = true, returnElement = false) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        const messageId = generateMessageId();
        messageElement.dataset.messageId = messageId;
        
        // Use a simple method to format the text
        const formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\n/g, '<br>'); // Newlines

        messageElement.innerHTML = `
            <div class="message-content">${formattedText}</div>
            ${sender === 'user' ? createMessageMenu(messageId) : ''}
        `;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;

        // Save chats after adding a message
        if (save) {
            saveChats();
        }

        if (returnElement) {
            return messageElement;
        }
    };

    // Toggle dropdown menu
    const toggleMenu = (messageId) => {
        // Close all other menus first
        document.querySelectorAll('.menu-dropdown.show').forEach(menu => {
            if (menu.id !== `menu-${messageId}`) {
                menu.classList.remove('show');
            }
        });
        
        const menu = document.getElementById(`menu-${messageId}`);
        if (menu) {
            menu.classList.toggle('show');
        }
    };

    // Close menus when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.message-menu')) {
            document.querySelectorAll('.menu-dropdown.show').forEach(menu => {
                menu.classList.remove('show');
            });
        }
    });

    // Delete message pair (user message + bot response)
    const deleteMessagePair = (messageId) => {
        const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageElement) return;

        // Find the next sibling (bot response)
        const botResponse = messageElement.nextElementSibling;
        
        // Animate and remove user message
        messageElement.style.transition = 'opacity 0.3s, transform 0.3s, max-height 0.3s';
        messageElement.style.opacity = '0';
        messageElement.style.transform = 'translateX(-20px)';
        
        // Animate and remove bot response if it exists
        if (botResponse && botResponse.classList.contains('bot-message')) {
            botResponse.style.transition = 'opacity 0.3s, transform 0.3s, max-height 0.3s';
            botResponse.style.opacity = '0';
            botResponse.style.transform = 'translateX(20px)';
        }
        
        setTimeout(() => {
            if (botResponse && botResponse.classList.contains('bot-message')) {
                botResponse.remove();
            }
            messageElement.remove();
            saveChats();
        }, 300);
    };

    // Edit a message (ChatGPT-style: edit and resubmit)
    const editMessage = (messageId) => {
        const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageElement) return;

        // Close the menu
        const menu = document.getElementById(`menu-${messageId}`);
        if (menu) menu.classList.remove('show');

        const contentEl = messageElement.querySelector('.message-content');
        const menuEl = messageElement.querySelector('.message-menu');
        
        // Get original text (strip HTML formatting)
        const originalHtml = contentEl.innerHTML;
        const originalText = contentEl.innerText || contentEl.textContent;

        // Hide menu while editing
        if (menuEl) menuEl.style.display = 'none';

        // Add editing class for styling
        messageElement.classList.add('editing');

        // Create edit interface
        const editContainer = document.createElement('div');
        editContainer.classList.add('edit-container');
        editContainer.innerHTML = `
            <textarea class="edit-textarea">${originalText}</textarea>
            <div class="edit-buttons">
                <button class="edit-cancel-btn" onclick="cancelEdit('${messageId}')">Cancel</button>
                <button class="edit-save-btn" onclick="saveAndResubmit('${messageId}')">Save & Submit</button>
            </div>
        `;

        // Store original content for cancel
        messageElement.dataset.originalContent = originalHtml;
        
        // Replace content with edit interface
        contentEl.innerHTML = '';
        contentEl.appendChild(editContainer);

        // Focus the textarea and auto-resize
        const textarea = editContainer.querySelector('.edit-textarea');
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';

        // Auto-resize on input
        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        });
    };

    // Save edit and resubmit to get new response (ChatGPT-style)
    const saveAndResubmit = async (messageId) => {
        const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageElement) return;

        const textarea = messageElement.querySelector('.edit-textarea');
        const newText = textarea.value.trim();

        if (newText === '') {
            deleteMessagePair(messageId);
            return;
        }

        // Find and remove the old bot response
        const oldBotResponse = messageElement.nextElementSibling;
        if (oldBotResponse && oldBotResponse.classList.contains('bot-message')) {
            oldBotResponse.style.transition = 'opacity 0.2s';
            oldBotResponse.style.opacity = '0';
            setTimeout(() => oldBotResponse.remove(), 200);
        }

        // Update the user message content
        const contentEl = messageElement.querySelector('.message-content');
        const menuEl = messageElement.querySelector('.message-menu');
        const formattedText = newText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        
        contentEl.innerHTML = formattedText;
        if (menuEl) menuEl.style.display = '';
        messageElement.classList.remove('editing');
        delete messageElement.dataset.originalContent;

        // Show loading and resubmit
        setLoading(true);
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: newText }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                if (errorData.error_code === 'VALIDATION_ERROR' && errorData.details?.errors) {
                    const roastMessage = errorData.details.errors[0];
                    const cleanMessage = roastMessage.replace(/^body\.text:\s*(Value error,\s*)?/i, '');
                    addMessage(cleanMessage, 'bot');
                } else {
                    addMessage(errorData.error || 'The Oracle is momentarily confused...', 'bot');
                }
                return;
            }

            const data = await response.json();
            displayAnalysis(data);

        } catch (error) {
            console.error('Error:', error);
            addMessage(`⚡ By the ancient scrolls! The Oracle hath encountered a disturbance in the mystical forces.`, 'bot');
        } finally {
            setLoading(false);
            saveChats();
        }
    };

    // Cancel editing
    const cancelEdit = (messageId) => {
        const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
        if (!messageElement) return;

        const contentEl = messageElement.querySelector('.message-content');
        const menuEl = messageElement.querySelector('.message-menu');
        const originalContent = messageElement.dataset.originalContent;

        if (originalContent) {
            contentEl.innerHTML = originalContent;
        }
        if (menuEl) menuEl.style.display = '';
        messageElement.classList.remove('editing');
        delete messageElement.dataset.originalContent;
    };

    // Expose functions globally
    window.toggleMenu = toggleMenu;
    window.deleteMessagePair = deleteMessagePair;
    window.editMessage = editMessage;
    window.saveAndResubmit = saveAndResubmit;
    window.cancelEdit = cancelEdit;

    const handleSend = async () => {
        const text = userInput.value.trim();
        if (text === '') return;

        addMessage(text, 'user');
        userInput.value = '';
        userInput.style.height = 'auto'; // Reset height
        setLoading(true);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                // Check if it's a validation error (has the witty roast messages)
                if (errorData.error_code === 'VALIDATION_ERROR' && errorData.details?.errors) {
                    // Extract the roast message from the validation error
                    const roastMessage = errorData.details.errors[0];
                    // Remove the "body.text: Value error, " prefix if present
                    const cleanMessage = roastMessage.replace(/^body\.text:\s*(Value error,\s*)?/i, '');
                    addMessage(cleanMessage, 'bot');
                } else {
                    // For other errors, show the error message directly
                    addMessage(errorData.error || 'The Oracle is momentarily confused...', 'bot');
                }
                return;
            }

            const data = await response.json();
            displayAnalysis(data);

        } catch (error) {
            console.error('Error:', error);
            addMessage(`⚡ By the ancient scrolls! The Oracle hath encountered a disturbance in the mystical forces. Please ensure the backend server is running!`, 'bot');
        } finally {
            setLoading(false);
        }
    };

    const displayAnalysis = (data) => {
        const { prediction, advice } = data;

        // Determine risk color and emoji
        let riskColor = "green";
        let riskEmoji = "🟢";
        if (prediction.risk_level.includes("ELEVATED") || prediction.risk_level.includes("Moderate")) {
            riskColor = "orange";
            riskEmoji = "🟡";
        }
        if (prediction.risk_level.includes("HIGH") || prediction.risk_level.includes("High")) {
            riskColor = "red";
            riskEmoji = "🔴";
        }

        // Build probability distribution bars
        const probabilities = prediction.probabilities;
        const buildBar = (value) => {
            const filled = Math.round(value * 20);
            const empty = 20 - filled;
            return '█'.repeat(filled) + '░'.repeat(empty);
        };

        const healthyProb = probabilities['Healthy'] || probabilities[0] || 0;
        const stressedProb = probabilities['Stressed'] || probabilities[1] || 0;
        const burnoutProb = probabilities['Burnout'] || probabilities[2] || 0;

        let botResponse = `**🔮 THE ORACLE SPEAKS:**

**Thy Mental State:** ${prediction.label.toUpperCase()}
**Confidence:** ${Math.round(prediction.confidence * 100)}%
**Risk Level:** ${riskEmoji} <span style="color:${riskColor};">${prediction.risk_level}</span>

**📊 Probability Distribution:**
<code>Healthy:  [${buildBar(healthyProb)}] ${(healthyProb * 100).toFixed(1)}%
Stressed: [${buildBar(stressedProb)}] ${(stressedProb * 100).toFixed(1)}%
Burnout:  [${buildBar(burnoutProb)}] ${(burnoutProb * 100).toFixed(1)}%</code>

**📝 Summary:**
${advice.summary}

**Severity Score:** ${advice.severity_score}/10

**📋 TOP RECOMMENDATIONS:**`;
        
        // Handle recommendations that are objects with title, description, action_items
        let recommendationsHtml = '<ul>';
        advice.recommendations.forEach(rec => {
            if (typeof rec === 'object') {
                recommendationsHtml += `<li><strong>${rec.category} ${rec.title}</strong><br>${rec.description}`;
                if (rec.action_items && rec.action_items.length > 0) {
                    recommendationsHtml += '<ul>';
                    rec.action_items.slice(0, 2).forEach(item => {
                        recommendationsHtml += `<li>${item}</li>`;
                    });
                    recommendationsHtml += '</ul>';
                }
                recommendationsHtml += '</li>';
            } else {
                recommendationsHtml += `<li>${rec}</li>`;
            }
        });
        recommendationsHtml += '</ul>';
        botResponse += recommendationsHtml;

        if (advice.emergency_resources) {
            botResponse += `\n\n**🚨 Emergency Resources:**
If you are in crisis, please reach out:
- **National Suicide Prevention Lifeline:** 988
- **Crisis Text Line:** Text HOME to 741741`;
        }

        botResponse += `\n\n**${advice.quick_tip}**`;
        botResponse += `\n\n**📅 Follow-up:** ${advice.follow_up}`;

        addMessage(botResponse, 'bot');
    };

    // Create typing indicator element
    const createTypingIndicator = () => {
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('message', 'bot-message', 'typing-indicator');
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-content">
                <span class="typing-text">The Oracle is consulting the ancient scrolls</span>
                <div class="typing-dots">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            </div>
        `;
        return typingDiv;
    };

    const showTypingIndicator = () => {
        // Remove any existing typing indicator
        const existing = document.getElementById('typing-indicator');
        if (existing) existing.remove();
        
        const typingIndicator = createTypingIndicator();
        chatBox.appendChild(typingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const hideTypingIndicator = () => {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.style.opacity = '0';
            typingIndicator.style.transform = 'translateY(-10px)';
            setTimeout(() => typingIndicator.remove(), 200);
        }
    };

    const setLoading = (isLoading) => {
        spinner.style.display = isLoading ? 'block' : 'none';
        sendBtn.disabled = isLoading;
        userInput.disabled = isLoading;
        
        if (isLoading) {
            showTypingIndicator();
        } else {
            hideTypingIndicator();
        }
    };

    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });

    // Load saved chats on page load
    const savedChats = localStorage.getItem(STORAGE_KEY);
    if (savedChats && JSON.parse(savedChats).length > 0) {
        // Clear the default welcome message and load saved chats
        chatBox.innerHTML = '';
        loadChats();
    }
});