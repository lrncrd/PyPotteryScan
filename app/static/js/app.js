        // ==========================================================================
        // App Dialog System (Custom in-app replacement for Confirm, Alert, Prompt)
        // ==========================================================================
        let appDialogResolver = null;
        let currentDialogMode = 'confirm';

        function closeAppDialog(result) {
            const modal = document.getElementById('appDialogModal');
            const card = document.getElementById('appDialogCard');
            if (!modal) return;

            if (card) card.classList.add('closing');
            modal.classList.remove('show');

            setTimeout(() => {
                modal.classList.add('hidden');
                if (card) card.classList.remove('closing');
                if (appDialogResolver) {
                    const resolve = appDialogResolver;
                    appDialogResolver = null;
                    resolve(result);
                }
            }, 200);
        }

        function showAppDialog({
            mode = 'confirm', // 'confirm' | 'alert' | 'prompt'
            title = '',
            subtitle = '',
            message = '',
            defaultValue = '',
            placeholder = '',
            confirmText = 'Confirm',
            cancelText = 'Cancel',
            type = 'danger', // 'danger' | 'warning' | 'info' | 'primary' | 'success'
            icon = null
        }) {
            return new Promise((resolve) => {
                appDialogResolver = resolve;
                currentDialogMode = mode;

                const modal = document.getElementById('appDialogModal');
                const titleEl = document.getElementById('appDialogTitle');
                const subtitleEl = document.getElementById('appDialogSubtitle');
                const messageEl = document.getElementById('appDialogMessage');
                const iconWrap = document.getElementById('appDialogIconWrap');
                const iconEl = document.getElementById('appDialogIcon');
                const promptWrap = document.getElementById('appDialogPromptContainer');
                const promptInput = document.getElementById('appDialogPromptInput');
                const cancelBtn = document.getElementById('appDialogCancelBtn');
                const confirmBtn = document.getElementById('appDialogConfirmBtn');

                if (!modal) {
                    if (mode === 'confirm') resolve(window.confirm(message));
                    else if (mode === 'prompt') resolve(window.prompt(message, defaultValue));
                    else { window.alert(message); resolve(); }
                    return;
                }

                // Titles & text
                titleEl.textContent = title || (mode === 'confirm' ? 'Confirmation' : mode === 'prompt' ? 'Input' : 'Notice');
                if (subtitle) {
                    subtitleEl.textContent = subtitle;
                    subtitleEl.classList.remove('hidden');
                } else {
                    subtitleEl.classList.add('hidden');
                }
                messageEl.textContent = message;

                // Icons & color styling
                iconWrap.className = `app-dialog-icon-wrapper ${type}`;
                let iconClass = icon;
                if (!iconClass) {
                    if (type === 'danger') iconClass = 'bi-exclamation-triangle-fill';
                    else if (type === 'warning') iconClass = 'bi-exclamation-circle-fill';
                    else if (type === 'success') iconClass = 'bi-check-circle-fill';
                    else if (mode === 'prompt') iconClass = 'bi-pencil-square';
                    else if (type === 'primary') iconClass = 'bi-chat-square-dots-fill';
                    else iconClass = 'bi-info-circle-fill';
                }
                iconEl.className = `bi ${iconClass}`;

                // Confirm button styling
                confirmBtn.textContent = confirmText;
                confirmBtn.className = `btn ${type === 'danger' ? 'btn-danger' : 'btn-primary'} app-dialog-btn`;

                // Cancel button
                if (mode === 'alert') {
                    cancelBtn.classList.add('hidden');
                } else {
                    cancelBtn.textContent = cancelText;
                    cancelBtn.classList.remove('hidden');
                }

                // Prompt input
                if (mode === 'prompt') {
                    promptWrap.classList.remove('hidden');
                    promptInput.value = defaultValue || '';
                    promptInput.placeholder = placeholder || '';
                } else {
                    promptWrap.classList.add('hidden');
                }

                // Show modal
                modal.classList.remove('hidden');
                modal.offsetHeight;
                modal.classList.add('show');

                // Auto-focus
                if (mode === 'prompt') {
                    setTimeout(() => {
                        promptInput.focus();
                        promptInput.select();
                    }, 60);
                } else {
                    setTimeout(() => confirmBtn.focus(), 60);
                }
            });
        }

        function showConfirmDialog(options) {
            return showAppDialog({ mode: 'confirm', ...options });
        }

        function showAlertDialog(options) {
            if (typeof options === 'string') {
                options = { message: options };
            }
            return showAppDialog({ mode: 'alert', type: 'info', confirmText: 'OK', ...options });
        }

        function showPromptDialog(options) {
            return showAppDialog({ mode: 'prompt', type: 'primary', confirmText: 'Submit', ...options });
        }

        window.appConfirm = (message, options = {}) => {
            if (typeof options === 'string') options = { title: options };
            return showConfirmDialog({ message, ...options });
        };

        window.appAlert = (message, options = {}) => {
            if (typeof options === 'string') options = { title: options };
            return showAlertDialog({ message, ...options });
        };

        window.appPrompt = (message, defaultValue = '', options = {}) => {
            return showPromptDialog({ message, defaultValue, ...options });
        };

        // Wire dialog events
        document.addEventListener('DOMContentLoaded', () => {
            const confirmBtn = document.getElementById('appDialogConfirmBtn');
            const cancelBtn = document.getElementById('appDialogCancelBtn');
            const closeBtn = document.getElementById('appDialogCloseBtn');
            const modal = document.getElementById('appDialogModal');
            const promptInput = document.getElementById('appDialogPromptInput');

            if (confirmBtn) {
                confirmBtn.addEventListener('click', () => {
                    if (currentDialogMode === 'prompt') {
                        closeAppDialog(promptInput ? promptInput.value.trim() : '');
                    } else {
                        closeAppDialog(true);
                    }
                });
            }

            if (cancelBtn) {
                cancelBtn.addEventListener('click', () => {
                    closeAppDialog(currentDialogMode === 'prompt' ? null : false);
                });
            }

            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    closeAppDialog(currentDialogMode === 'prompt' ? null : false);
                });
            }

            if (modal) {
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        closeAppDialog(currentDialogMode === 'prompt' ? null : false);
                    }
                });
            }

            if (promptInput) {
                promptInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        closeAppDialog(promptInput.value.trim());
                    }
                });
            }
        });

        // Intercept global alert to use styled in-app modal
        window.alert = function(msg) {
            const str = String(msg);
            const isError = /error|failed|failure|unable|cannot|could not/i.test(str);
            const isWarning = /warning|please|must|required|missing|invalid/i.test(str);
            const isSuccess = /success|completed|downloaded|saved/i.test(str);
            const type = isError ? 'danger' : isWarning ? 'warning' : isSuccess ? 'success' : 'info';
            const title = isError ? 'Error' : isWarning ? 'Notice' : isSuccess ? 'Success' : 'Notice';
            return showAlertDialog({
                title,
                message: str,
                type,
                confirmText: 'OK'
            });
        };

        // Model selection state
        let selectedModel = 'NONE'; // Default selection, upgraded to FP4/GLM once CUDA is detected
        let cudaAvailable = false;
        let modelSelectionDefaulted = false; // Ensures the auto-selected default only applies once
        // True while the user has backed out of an error to pick a different model. The status
        // poller keeps running through the 'error' stage (see checkServer) so a retry can resume
        // automatically; this flag stops it from re-showing the error panel over that choice.
        let choosingAlternateModel = false;

        // Highlights whichever model card matches the current `selectedModel`, clearing the
        // others. Shared by card clicks and by "Choose a Different Model" (whose selectedModel
        // may have just been synced from the server, e.g. a saved 'FP4' pick the card UI was
        // never shown for).
        function syncCardSelectionUI() {
            const cardIdByModel = { FP4: 'fp4Card', GLM: 'glmCard', NONE: 'noneCard' };
            Object.entries(cardIdByModel).forEach(([modelId, elementId]) => {
                document.getElementById(elementId)?.classList.toggle('selected', modelId === selectedModel);
            });
        }

        // Initialize model selection UI
        function initModelSelection() {
            const fp4Card = document.getElementById('fp4Card');
            const glmCard = document.getElementById('glmCard');
            const noneCard = document.getElementById('noneCard');
            const confirmBtn = document.getElementById('confirmModelBtn');

            // Card selection handlers
            fp4Card.onclick = () => {
                if (cudaAvailable) {
                    selectedModel = 'FP4';
                    syncCardSelectionUI();
                }
            };

            glmCard.onclick = () => {
                selectedModel = 'GLM';
                syncCardSelectionUI();
            };

            noneCard.onclick = () => {
                selectedModel = 'NONE';
                syncCardSelectionUI();
            };

            // Confirm button handler
            confirmBtn.onclick = async () => {
                try {
                    confirmBtn.disabled = true;
                    confirmBtn.innerHTML = '<i class="bi bi-arrow-repeat spin mr-1"></i> Starting download...';
                    await submitModelSelection(selectedModel);
                } catch (err) {
                    alert('Network error: ' + err.message);
                } finally {
                    confirmBtn.disabled = false;
                    confirmBtn.innerHTML = '<i class="bi bi-cloud-arrow-down mr-1"></i> Download Selected Model';
                }
            };
        }

        // Ask the server to (re)download/select an OCR model, then swap the splash screen
        // back into its progress view. Shared by the initial model-selection confirm button
        // and by the error-recovery actions (retry / continue without OCR).
        async function submitModelSelection(modelId) {
            const response = await fetch('/select_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId })
            });
            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Unknown error');
            }

            selectedModel = modelId;
            choosingAlternateModel = false;
            document.getElementById('modelSelectionUI').classList.add('hidden');
            document.getElementById('modelSelectionUI').style.display = 'none';
            document.getElementById('splashErrorActions').classList.add('hidden');
            document.getElementById('splashErrorActions').style.display = 'none';
            document.getElementById('progressBarContainer').classList.remove('hidden');
            document.getElementById('progressBarContainer').style.display = '';
            document.getElementById('loadingSpinner').classList.remove('hidden');
            document.getElementById('loadingSpinner').style.display = '';
            document.getElementById('splashProgress').style.background = 'linear-gradient(90deg, #c2410c, #ea580c)';
        }

        // Wire the error-recovery actions shown when a download fails: retry the same
        // model, switch to a different one, or skip OCR entirely and continue.
        function initErrorRecoveryActions() {
            const retryBtn = document.getElementById('retryDownloadBtn');
            const changeModelBtn = document.getElementById('changeModelBtn');
            const continueNoOcrBtn = document.getElementById('continueNoOcrBtn');

            async function attempt(btn, modelId, busyLabel) {
                const originalLabel = btn.innerHTML;
                try {
                    btn.disabled = true;
                    btn.innerHTML = `<i class="bi bi-arrow-repeat spin mr-1"></i> ${busyLabel}`;
                    await submitModelSelection(modelId);
                } catch (err) {
                    alert('Network error: ' + err.message);
                    btn.disabled = false;
                    btn.innerHTML = originalLabel;
                }
            }

            retryBtn.onclick = () => attempt(retryBtn, selectedModel, 'Retrying...');
            continueNoOcrBtn.onclick = () => attempt(continueNoOcrBtn, 'NONE', 'Continuing...');

            changeModelBtn.onclick = () => {
                choosingAlternateModel = true;
                document.getElementById('splashErrorActions').classList.add('hidden');
                document.getElementById('splashErrorActions').style.display = 'none';
                document.getElementById('modelSelectionUI').classList.remove('hidden');
                document.getElementById('modelSelectionUI').style.display = '';
                document.getElementById('progressBarContainer').classList.add('hidden');
                document.getElementById('progressBarContainer').style.display = 'none';
                document.getElementById('loadingSpinner').classList.add('hidden');
                document.getElementById('loadingSpinner').style.display = 'none';
                document.getElementById('splashTitle').textContent = 'Model Setup';
                document.getElementById('splashLog').innerHTML = '<p style="color: var(--text-dim);">Select a different model to download</p>';
                // Highlight whichever model actually failed (synced from the server just
                // before this panel was shown), not whatever the cards defaulted to.
                syncCardSelectionUI();
            };
        }

        function formatBytes(bytes) {
            if (!bytes) return '0 MB';
            const mb = bytes / (1024 * 1024);
            return mb < 1024 ? `${mb.toFixed(0)} MB` : `${(mb / 1024).toFixed(2)} GB`;
        }

        // Renders one real progress bar per entry in /loading_status's `downloads` list (the
        // OCR model and the Qwen parsing model), each driven by actual bytes on disk rather
        // than a fixed checkpoint, so the bar keeps moving for as long as the download does.
        // ONLY shown during active 'downloading' stage. When models already exist on disk or
        // when loading into memory, download bars are hidden.
        function renderDownloadBars(downloads, stage) {
            const container = document.getElementById('downloadBarsContainer');
            if (!container) return;

            if (!downloads || downloads.length === 0 || stage !== 'downloading') {
                container.classList.add('hidden');
                container.style.display = 'none';
                container.innerHTML = '';
                return;
            }
            container.classList.remove('hidden');
            container.style.display = 'block';
            container.innerHTML = downloads.map(d => {
                const pct = d.status === 'done' ? 100 : (d.progress || 0);
                const isError = d.status === 'error';
                const barBackground = isError ? 'var(--danger-color)' : 'linear-gradient(90deg, #c2410c, #ea580c)';
                let statusLabel;
                if (d.status === 'pending') statusLabel = 'Waiting…';
                else if (d.status === 'done') statusLabel = 'Done';
                else if (isError) statusLabel = 'Failed';
                else statusLabel = d.bytes_total ? `${formatBytes(d.bytes_done)} / ${formatBytes(d.bytes_total)}` : formatBytes(d.bytes_done);
                return `
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--text-dim); margin-bottom: 3px;">
                            <span>${d.name}</span>
                            <span>${statusLabel}</span>
                        </div>
                        <div class="splash-progress" style="margin: 0;">
                            <div class="splash-progress-bar" style="width: ${pct}%; background: ${barBackground};">${d.status === 'downloading' || d.status === 'done' ? pct + '%' : ''}</div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Check server and poll loading status
        async function checkServer() {
            const log = document.getElementById('splashLog');
            const progressBar = document.getElementById('splashProgress');
            const titleElement = document.getElementById('splashTitle');
            const modelSelectionUI = document.getElementById('modelSelectionUI');
            const progressBarContainer = document.getElementById('progressBarContainer');
            const loadingSpinner = document.getElementById('loadingSpinner');

            log.innerHTML = '<p><i class="bi bi-plug mr-1"></i> Connecting to OCR server...</p>';

            initModelSelection();
            initErrorRecoveryActions();

            // Poll loading status every 500ms
            const pollInterval = setInterval(async () => {
                try {
                    const response = await fetch('/loading_status');
                    const status = await response.json();

                    // Update progress bar
                    const progress = status.progress || 0;
                    progressBar.style.width = progress + '%';
                    progressBar.textContent = progress + '%';
                    renderDownloadBars(status.downloads, status.stage);

                    // Update title and log based on status
                    if (status.stage === 'starting') {
                        titleElement.textContent = 'Initializing...';
                        progressBarContainer.classList.remove('hidden');
                        progressBarContainer.style.display = '';
                        loadingSpinner.classList.remove('hidden');
                        loadingSpinner.style.display = '';
                        log.innerHTML = `<p style="color: var(--primary);"><i class="bi bi-hourglass-split mr-1"></i> ${status.message}</p>`;
                    } else if (status.stage === 'checking') {
                        titleElement.textContent = 'Checking Models...';
                        progressBarContainer.classList.remove('hidden');
                        progressBarContainer.style.display = '';
                        loadingSpinner.classList.remove('hidden');
                        loadingSpinner.style.display = '';
                        log.innerHTML = `<p style="color: var(--primary);"><i class="bi bi-search mr-1"></i> ${status.message}</p>`;
                    } else if (status.stage === 'model_selection') {
                        // Show model selection UI
                        titleElement.textContent = 'Model Setup';
                        progressBarContainer.classList.add('hidden');
                        progressBarContainer.style.display = 'none';
                        loadingSpinner.classList.add('hidden');
                        loadingSpinner.style.display = 'none';
                        modelSelectionUI.classList.remove('hidden');
                        modelSelectionUI.style.display = '';

                        // Check CUDA availability and update FP4 card
                        try {
                            const modelsResp = await fetch('/available_models');
                            const modelsData = await modelsResp.json();
                            cudaAvailable = modelsData.cuda_available;

                            const fp4Card = document.getElementById('fp4Card');
                            const fp4Req = document.getElementById('fp4Requirement');

                            if (!cudaAvailable) {
                                fp4Card.style.opacity = '0.5';
                                fp4Card.style.cursor = 'not-allowed';
                                fp4Req.innerHTML = '<i class="bi bi-x-circle mr-1"></i> NVIDIA GPU not detected';
                                fp4Req.style.color = 'var(--danger-color)';
                            } else {
                                // GPU present: FP4 is available but GLM-OCR stays the default
                                // selection below, so invite the switch instead of assuming it.
                                fp4Req.innerHTML = '<i class="bi bi-check-circle-fill mr-1"></i> NVIDIA GPU detected — select this instead for higher quality';
                                fp4Req.style.color = 'var(--teal)';
                            }

                            // Only apply the auto-selected default once: this block re-runs on
                            // every 500ms status poll while we stay in the model_selection stage,
                            // and would otherwise stomp on the user's manual card click.
                            if (!modelSelectionDefaulted) {
                                modelSelectionDefaulted = true;
                                // GLM-OCR is the default regardless of hardware: it's small,
                                // fast, and works everywhere. Users with an NVIDIA GPU are
                                // invited (via the FP4 card's badge above) to switch to FP4
                                // for higher quality, but nothing is auto-selected for them.
                                // The GLM card already carries the "selected" class in the
                                // markup, so no DOM change is needed here.
                                selectedModel = 'GLM';
                            }
                        } catch (e) {
                            console.error('Error checking models:', e);
                        }

                        log.innerHTML = `<p style="color: var(--text-dim);">First time setup - please select a model to download</p>`;
                    } else if (status.stage === 'downloading') {
                        titleElement.textContent = 'Downloading Models...';
                        // The per-model bars below carry the real progress now; this single
                        // legacy bar would otherwise just sit at its last coarse checkpoint.
                        progressBarContainer.classList.add('hidden');
                        progressBarContainer.style.display = 'none';
                        loadingSpinner.classList.remove('hidden');
                        loadingSpinner.style.display = '';
                        const modelName = status.download_model || 'Model';
                        log.innerHTML = `<p style="color: var(--warning-color);"><i class="bi bi-cloud-arrow-down mr-1"></i> Downloading ${modelName}...</p>`;
                        log.innerHTML += `<p style="color: var(--dark-fg-muted); font-size: 12px;">This may take several minutes depending on your internet connection...</p>`;
                    } else if (status.stage === 'ready_to_load') {
                        progressBarContainer.classList.remove('hidden');
                        progressBarContainer.style.display = '';
                        loadingSpinner.classList.remove('hidden');
                        loadingSpinner.style.display = '';
                        titleElement.textContent = 'Loading Models...';
                        log.innerHTML = `<p style="color: var(--teal);"><i class="bi bi-check-circle-fill mr-1"></i> Models downloaded successfully!</p>`;
                        log.innerHTML += `<p style="color: var(--primary);"><i class="bi bi-box-seam mr-1"></i> Loading into memory...</p>`;
                    } else if (status.stage === 'loading') {
                        progressBarContainer.classList.remove('hidden');
                        progressBarContainer.style.display = '';
                        loadingSpinner.classList.remove('hidden');
                        loadingSpinner.style.display = '';
                        titleElement.textContent = 'Loading Models...';
                        log.innerHTML = `<p style="color: var(--warning-color);"><i class="bi bi-box-seam mr-1"></i> ${status.message}</p>`;
                        log.innerHTML += `<p style="color: var(--teal);"><i class="bi bi-hourglass-split mr-1"></i> Progress: ${progress}%</p>`;
                    } else if (status.stage === 'finalizing') {
                        progressBarContainer.classList.remove('hidden');
                        progressBarContainer.style.display = '';
                        loadingSpinner.classList.remove('hidden');
                        loadingSpinner.style.display = '';
                        titleElement.textContent = 'Almost Ready...';
                        log.innerHTML = `<p style="color: var(--teal);"><i class="bi bi-stars mr-1"></i> ${status.message}</p>`;
                    } else if (status.stage === 'ready') {
                        clearInterval(pollInterval);

                        titleElement.textContent = 'Ready!';
                        progressBarContainer.classList.remove('hidden');
                        progressBarContainer.style.display = '';
                        loadingSpinner.classList.add('hidden');
                        loadingSpinner.style.display = 'none';
                        progressBar.style.width = '100%';
                        progressBar.textContent = '100%';

                        // Get final health check
                        const healthResponse = await fetch('/health');
                        const healthData = await healthResponse.json();

                        window.ocrAvailable = healthData.ocr_available;
                        window.qwenAvailable = healthData.qwen_available;

                        log.innerHTML = `<p style="color: var(--teal);"><i class="bi bi-check-circle-fill mr-1"></i> Server online: ${healthData.status}</p>`;
                        log.innerHTML += `<p style="color: var(--teal);"><i class="bi bi-check-circle-fill mr-1"></i> Model: ${healthData.model}</p>`;
                        log.innerHTML += `<p style="color: var(--teal);"><i class="bi bi-check-circle-fill mr-1"></i> Device: ${healthData.device}</p>`;
                        log.innerHTML += `<p style="color: var(--teal);"><i class="bi bi-check-circle-fill mr-1"></i> CUDA: ${healthData.cuda_available ? 'Available' : 'CPU Mode'}</p>`;
                        log.innerHTML += '<p style="color: var(--teal); font-weight: bold;"><i class="bi bi-check-circle-fill mr-1"></i> Ready to start!</p>';

                        setTimeout(() => {
                            document.getElementById('splashScreen').style.display = 'none';
                        }, 1500);
                    } else if (status.stage === 'error' && !choosingAlternateModel) {
                        // Keep polling (unlike the 'ready' terminal stage): a retry/continue
                        // action below moves the backend out of 'error' on its own, and this
                        // same interval needs to notice that without a manual restart. Skipped
                        // while choosingAlternateModel is true so it doesn't fight the user's
                        // "Choose a Different Model" panel switch every 500ms.
                        titleElement.textContent = 'Setup Paused';
                        loadingSpinner.classList.add('hidden');
                        loadingSpinner.style.display = 'none';
                        progressBarContainer.classList.add('hidden');
                        progressBarContainer.style.display = 'none';
                        progressBar.style.background = 'var(--danger-color)';
                        log.innerHTML = `<p style="color: var(--danger-color);"><i class="bi bi-x-circle-fill mr-1"></i> ${status.message}</p>`;
                        document.getElementById('splashErrorActions').classList.remove('hidden');
                        document.getElementById('splashErrorActions').style.display = 'flex';

                        // A model chosen on a previous run (saved to disk) auto-starts its
                        // download on boot without ever going through the card UI below, so
                        // the client-side `selectedModel` is still its unset default here.
                        // Sync it from the server before Retry can use it.
                        try {
                            const modelsResp = await fetch('/available_models');
                            const modelsData = await modelsResp.json();
                            if (modelsData.selected) {
                                selectedModel = modelsData.selected;
                            }
                        } catch (e) {
                            console.error('Error syncing selected model:', e);
                        }
                    }
                } catch (err) {
                    // Server not ready yet, keep polling
                }
            }, 500);

            // Timeout after 5 minutes (to account for large downloads)
            setTimeout(() => {
                clearInterval(pollInterval);
                log.innerHTML += '<p style="color: #b91c1c;"><i class="bi bi-x-circle-fill mr-1"></i> Timeout: Model loading took too long</p>';
            }, 300000);
        }

        checkServer();

        // ==========================================
        // PROJECT MANAGEMENT
        // ==========================================

        let currentProject = null;

        // Load projects on page load
        async function loadProjects() {
            try {
                const response = await fetch('/api/project/list');
                const data = await response.json();

                if (data.success) {
                    renderProjects(data.projects);
                }
            } catch (error) {
                console.error('Error loading projects:', error);
            }
        }

        // Render projects grid
        function renderProjects(projects) {
            const grid = document.getElementById('projectsGrid');
            const emptyState = document.getElementById('emptyProjectsState');

            if (projects.length === 0) {
                grid.classList.add('hidden');
                emptyState.classList.remove('hidden');
                return;
            }

            grid.classList.remove('hidden');
            emptyState.classList.add('hidden');

            grid.innerHTML = projects.map(project => {
                const created = new Date(project.created_at).toLocaleDateString();
                const modified = new Date(project.last_modified).toLocaleDateString();
                const status = project.workflow_status;

                // Calculate completion percentage
                const totalImages = status.images_count || 0;
                const annotatedImages = status.annotations_completed || 0;
                const completionPercent = totalImages > 0 ? Math.round((annotatedImages / totalImages) * 100) : 0;

                return `
                    <div class="project-card p-4 hover:shadow-md transition cursor-pointer" data-project-id="${project.project_id}">
                        <div class="flex justify-between items-start mb-3">
                            <h3 class="text-base font-bold text-stone-800 truncate flex-1">${project.project_name}</h3>
                            <button class="delete-project-btn" data-project-id="${project.project_id}" onclick="event.stopPropagation();" title="Delete Project">
                                <i class="bi bi-trash3 pointer-events-none"></i>
                            </button>
                        </div>
                        
                        ${project.description ? `<p class="text-xs text-stone-600 mb-3 line-clamp-2">${project.description}</p>` : ''}
                        
                        <div class="space-y-1 text-xs text-stone-600 mb-3">
                            <div><i class="bi bi-calendar3 mr-1 text-stone-400"></i> Created: ${created}</div>
                            <div><i class="bi bi-clock-history mr-1 text-stone-400"></i> Modified: ${modified}</div>
                            <div class="flex items-center gap-2">
                                <span><i class="bi bi-images mr-1 text-stone-400"></i> Images: ${totalImages}</span>
                                ${annotatedImages > 0 ? `<span class="text-teal-700 font-semibold"><i class="bi bi-check2-circle mr-1"></i> Annotated: ${annotatedImages}</span>` : ''}
                            </div>
                        </div>
                        
                        ${totalImages > 0 ? `
                            <div class="mb-3">
                                <div class="flex justify-between text-xs text-stone-600 mb-1">
                                    <span>Progress</span>
                                    <span class="font-mono">${completionPercent}%</span>
                                </div>
                                <div class="w-full bg-stone-200 rounded-full h-2 overflow-hidden">
                                    <div class="h-2 rounded-full transition-all" style="width: ${completionPercent}%; background: linear-gradient(90deg, #c2410c, #ea580c);"></div>
                                </div>
                            </div>
                        ` : ''}
                        
                        <div class="flex gap-2">
                            <button class="open-project-btn btn btn-primary flex-1 py-1.5 text-xs" data-project-id="${project.project_id}">
                                Open
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

            // Add event listeners
            grid.querySelectorAll('.project-card').forEach(card => {
                card.addEventListener('click', (e) => {
                    if (!e.target.closest('.delete-project-btn') && !e.target.closest('.open-project-btn')) {
                        const projectId = card.dataset.projectId;
                        openProject(projectId);
                    }
                });
            });

            grid.querySelectorAll('.open-project-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const projectId = btn.dataset.projectId;
                    openProject(projectId);
                });
            });

            grid.querySelectorAll('.delete-project-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const projectId = btn.dataset.projectId;

                    const confirmed = await showConfirmDialog({
                        title: 'Delete Project',
                        message: 'Are you sure you want to delete this project? This action cannot be undone and all associated images and annotations will be permanently removed.',
                        confirmText: 'Delete Project',
                        cancelText: 'Cancel',
                        type: 'danger',
                        icon: 'bi-trash3-fill'
                    });

                    if (confirmed) {
                        await deleteProject(projectId);
                    }
                });
            });
        }

        // Open project
        async function openProject(projectId) {
            try {
                // Show loading overlay
                const overlay = document.getElementById('loadTabLoadingOverlay');
                const loadingText = document.getElementById('loadTabLoadingText');
                overlay.classList.remove('hidden');
                loadingText.textContent = 'Opening project...';

                const response = await fetch(`/api/project/${projectId}`);
                const data = await response.json();

                if (data.success) {
                    currentProject = data.project;
                    console.log('Opened project:', currentProject);

                    // Enable Load tab
                    const loadTab = document.querySelector('.tab-btn[data-tab="load"]');
                    loadTab.disabled = false;

                    // Switch to Load tab
                    switchTab('load');

                    loadingText.textContent = 'Loading images and thumbnails...';

                    // Load project images and annotations
                    await loadProjectImagesWithAnnotations(projectId);

                    // Detect project state and enable appropriate tabs
                    await detectAndRestoreProjectState(projectId);

                    // Hide loading overlay
                    overlay.classList.add('hidden');
                }
            } catch (error) {
                console.error('Error opening project:', error);
                document.getElementById('loadTabLoadingOverlay').classList.add('hidden');
                alert('Failed to open project');
            }
        }

        // Load project images and check for existing annotations
        async function loadProjectImagesWithAnnotations(projectId) {
            const imagesList = document.getElementById('imagesList');
            let loadedCount = 0;

            try {
                const response = await fetch(`/api/project/${projectId}/images`);
                const data = await response.json();

                if (data.success && data.images.length > 0) {
                    state.images = [];
                    const annotationStatuses = {}; // Track which images have annotations
                    const totalImages = data.images.length;

                    // Load all images and check for annotations
                    for (const imageName of data.images) {
                        const imageUrl = `/api/project/${projectId}/image/${encodeURIComponent(imageName)}`;
                        const thumbUrl = `/api/project/${projectId}/image/${encodeURIComponent(imageName)}?thumbnail=true`;

                        state.images.push({
                            name: imageName,
                            url: imageUrl,
                            thumbUrl: thumbUrl
                        });

                        // Check if annotations exist for this image
                        const annotationData = await loadAnnotationsFromProject(imageName);
                        if (annotationData && annotationData.drawings && annotationData.drawings.length > 0) {
                            annotationStatuses[imageName] = annotationData.drawings.length;
                            // Pre-load into state
                            state.annotations[imageName] = annotationData;
                        } else {
                            annotationStatuses[imageName] = 0;
                        }

                        // Update progress in overlay
                        loadedCount++;
                        const loadingText = document.getElementById('loadTabLoadingText');
                        if (loadingText) {
                            loadingText.textContent = `Loading ${loadedCount} of ${totalImages} images...`;
                        }
                    }

                    console.log('Loaded', state.images.length, 'images');
                    console.log('Annotation statuses:', annotationStatuses);

                    // Note: Cleaned drawings will be loaded on-demand when switching to Clean tab

                    // Count total annotations
                    const totalAnnotations = Object.values(annotationStatuses).reduce((sum, count) => sum + count, 0);
                    const imagesWithAnnotations = Object.values(annotationStatuses).filter(count => count > 0).length;

                    // Update project workflow status
                    if (currentProject) {
                        currentProject.workflow_status.annotations_completed = imagesWithAnnotations;
                        await updateProjectWorkflowStatus({
                            annotations_completed: imagesWithAnnotations,
                            total_drawings: totalAnnotations
                        });
                    }

                    // Show folder info workstation, hide empty prompt
                    const folderInfo = document.getElementById('folderInfo');
                    const emptyState = document.getElementById('emptyLoadState');
                    if (folderInfo) folderInfo.classList.remove('hidden');
                    if (emptyState) emptyState.classList.add('hidden');

                    const folderNameEl = document.getElementById('folderName');
                    if (folderNameEl) folderNameEl.textContent = currentProject.project_name;

                    const imageCountEl = document.getElementById('imageCount');
                    if (imageCountEl) imageCountEl.textContent = data.count;

                    const imageBadge = document.getElementById('imageBadge');
                    if (imageBadge) {
                        imageBadge.textContent = `${data.count} plate${data.count === 1 ? '' : 's'}`;
                        imageBadge.classList.remove('hidden');
                    }

                    const annotatedSummaryEl = document.getElementById('annotatedSummaryText');
                    if (annotatedSummaryEl) {
                        annotatedSummaryEl.textContent = `${imagesWithAnnotations} of ${data.count} annotated`;
                    }

                    const catalogStatusEl = document.getElementById('catalogStatusText');
                    if (catalogStatusEl) {
                        catalogStatusEl.textContent = imagesWithAnnotations === data.count
                            ? 'All plates annotated'
                            : `${data.count - imagesWithAnnotations} plate${(data.count - imagesWithAnnotations) === 1 ? '' : 's'} awaiting annotation`;
                    }

                    // Display thumbnails with annotation indicators
                    renderThumbnailsWithAnnotations(projectId, data.images, annotationStatuses);
                } else {
                    // Empty project (0 images)
                    const folderInfo = document.getElementById('folderInfo');
                    const emptyState = document.getElementById('emptyLoadState');
                    const imageBadge = document.getElementById('imageBadge');
                    if (imageBadge) imageBadge.classList.add('hidden');
                    if (folderInfo) folderInfo.classList.add('hidden');
                    if (emptyState) emptyState.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Error loading project images:', error);
                const imagesList = document.getElementById('imagesList');
                if (imagesList) {
                    imagesList.innerHTML = `
                        <div class="col-span-full flex flex-col items-center justify-center py-12">
                            <p class="text-red-600 text-lg font-semibold flex items-center justify-center gap-2"><i class="bi bi-exclamation-circle-fill"></i> Error loading images</p>
                            <p class="text-gray-500 text-sm mt-2">${error.message}</p>
                        </div>
                    `;
                }
            }
        }

        // Render thumbnails with visual indicators for annotations
        function renderThumbnailsWithAnnotations(projectId, images, annotationStatuses) {
            const imagesList = document.getElementById('imagesList');
            if (!imagesList) return;

            imagesList.innerHTML = images.map((img, idx) => {
                const hasAnnotations = (annotationStatuses[img] || 0) > 0;
                const safeImg = encodeURIComponent(img);

                return `
                    <div class="thumbnail relative cursor-pointer ${hasAnnotations ? 'annotated' : ''}" data-index="${idx}" title="${img}">
                        <div class="h-36 bg-stone-50 flex items-center justify-center p-2.5 overflow-hidden border-b border-[#f0ede6]">
                            <img src="/api/project/${projectId}/image/${safeImg}?thumbnail=true" class="max-h-full max-w-full object-contain" loading="lazy" alt="${img}" onerror="if(!this.dataset.retry){this.dataset.retry='1';this.src='/api/project/${projectId}/image/${safeImg}';}">
                        </div>
                        <div class="px-3 py-2 bg-[#fdfcfb] flex items-center justify-between text-xs">
                            <span class="font-mono text-stone-700 truncate font-medium text-[11px]">${img}</span>
                            <span class="text-stone-400 text-[10px] shrink-0 font-semibold ml-1">#${idx + 1}</span>
                        </div>
                    </div>
                `;
            }).join('');

            // Add click handlers
            const thumbnails = imagesList.querySelectorAll('.thumbnail');
            thumbnails.forEach((thumb) => {
                thumb.addEventListener('click', () => {
                    const idx = parseInt(thumb.dataset.index);
                    state.currentImageIndex = idx;
                    loadImageForAnnotation();
                    enableTab('annotate');
                    switchTab('annotate');
                });
            });
        }

        // Load cleaned drawings from project
        async function loadCleanedDrawingsFromProject(projectId) {
            try {
                console.log('[Clean] Loading cleaned drawings from project...');

                // Get list of all cleaned drawings from the cleaned_drawings folder
                const response = await fetch(`/api/project/${projectId}/images?folder=cleaned_drawings`);
                const data = await response.json();

                if (data.success && data.images && data.images.length > 0) {
                    console.log(`   Found ${data.images.length} cleaned drawings`);

                    // Load each cleaned drawing
                    for (const filename of data.images) {
                        // Extract key from filename (e.g., "1342_004_d1_cleaned.png" -> "1342_004.jpg_d1")
                        const match = filename.match(/^(.+)_d(\d+)_cleaned\./);
                        if (match) {
                            const baseImgName = match[1];
                            const drawingNum = match[2];

                            // Find original image name (might have different extension)
                            const originalImg = state.images.find(img => img.name.startsWith(baseImgName));
                            if (originalImg) {
                                const key = `${originalImg.name}_d${drawingNum}`;

                                // Make sure full-res image is loaded for this image
                                if (!state.fullResImages[originalImg.name]) {
                                    console.log(`   Loading full-res image for: ${originalImg.name}`);
                                    const imgElement = new Image();
                                    imgElement.crossOrigin = 'anonymous';

                                    await new Promise((resolve, reject) => {
                                        imgElement.onload = () => {
                                            state.fullResImages[originalImg.name] = imgElement;
                                            console.log(`   [Clean] Full-res image loaded: ${originalImg.name}`);
                                            resolve();
                                        };
                                        imgElement.onerror = () => {
                                            console.error(`   [Clean] Failed to load image: ${originalImg.name}`);
                                            reject();
                                        };
                                        imgElement.src = `/api/project/${projectId}/image/${encodeURIComponent(originalImg.name)}`;
                                    });
                                }

                                // Load the cleaned image
                                const cleanedImgUrl = `/api/project/${projectId}/image/${encodeURIComponent(filename)}?folder=cleaned_drawings`;

                                // Fetch as data URL
                                const imgResponse = await fetch(cleanedImgUrl);
                                const blob = await imgResponse.blob();
                                const reader = new FileReader();

                                await new Promise((resolve) => {
                                    reader.onloadend = () => {
                                        if (!state.cleanedDrawings[key]) {
                                            state.cleanedDrawings[key] = {};
                                        }
                                        state.cleanedDrawings[key].cleaned = true;
                                        state.cleanedDrawings[key].imageData = reader.result;
                                        console.log(`   [Clean] Loaded cleaned drawing: ${key}`);
                                        resolve();
                                    };
                                    reader.readAsDataURL(blob);
                                });
                            } else {
                                console.warn(`   [Clean] Could not find original image for: ${filename}`);
                            }
                        }
                    }

                    console.log('[Clean] All cleaned drawings loaded');
                } else {
                    console.log('[Clean] No cleaned drawings found in project (this is normal for new projects)');
                }
            } catch (error) {
                console.error('[Clean] Error loading cleaned drawings:', error);
                // Don't throw - just log the error and continue
            }
        }

        // Detect project state and restore/enable tabs accordingly
        async function detectAndRestoreProjectState(projectId) {
            try {
                console.log('[State] Detecting project state...');

                // Check what exists in the project
                const hasAnnotations = Object.keys(state.annotations).length > 0;
                const hasCroppedDrawings = await checkFolderHasFiles(projectId, 'cropped_drawings');
                const hasCleanedDrawings = await checkFolderHasFiles(projectId, 'cleaned_drawings');

                // Check OCR results using dedicated endpoint (not folder check)
                const hasOCRResults = await checkOCRResultsExist(projectId);

                console.log('Project state:', {
                    hasAnnotations,
                    hasCroppedDrawings,
                    hasOCRResults,
                    hasCleanedDrawings
                });

                let restoredFeatures = [];

                // Always enable Annotate if we have images
                if (state.images.length > 0) {
                    enableTab('annotate');
                }

                // If we have annotations, enable Process tab
                if (hasAnnotations) {
                    enableTab('process');
                    restoredFeatures.push('annotations');

                    // If cropped drawings exist, pre-load them
                    if (hasCroppedDrawings) {
                        console.log('[State] Found existing cropped drawings, loading...');
                        await loadExistingCroppedDrawings(projectId);
                        restoredFeatures.push('cropped drawings');
                    }
                }

                // If we have OCR results, enable Review and Export tabs
                if (hasOCRResults) {
                    console.log('[State] Found existing OCR results, loading...');
                    await loadExistingOCRResults(projectId);
                    enableTab('review');
                    enableTab('export');
                    restoredFeatures.push('OCR results');
                }

                // If we have cleaned drawings, enable Clean tab
                if (hasCleanedDrawings || hasCroppedDrawings) {
                    enableTab('clean');
                    console.log('[State] Clean tab enabled');
                }

                if (restoredFeatures.length > 0) {
                    console.log(`[State] Project restored with: ${restoredFeatures.join(', ')}`);
                    // Show a brief notification
                    const statusText = `Project restored: ${restoredFeatures.join(', ')}`;
                    console.log(`[State] ${statusText}`);
                }

                console.log('[State] Project state restored');

            } catch (error) {
                console.error('Error detecting project state:', error);
            }
        }

        // Helper: Check if a folder has files
        async function checkFolderHasFiles(projectId, folderType) {
            try {
                const response = await fetch(`/api/project/${projectId}/images?folder=${folderType}`);
                const data = await response.json();
                return data.success && data.images && data.images.length > 0;
            } catch (error) {
                return false;
            }
        }

        // Helper: Check if OCR results exist using dedicated endpoint
        async function checkOCRResultsExist(projectId) {
            try {
                const response = await fetch(`/api/project/${projectId}/ocr_results`);
                const data = await response.json();
                return data.success && data.results && data.results.length > 0;
            } catch (error) {
                return false;
            }
        }

        // Load existing cropped drawings (just metadata, images loaded on demand)
        async function loadExistingCroppedDrawings(projectId) {
            try {
                const response = await fetch(`/api/project/${projectId}/images?folder=cropped_drawings`);
                const data = await response.json();

                if (data.success && data.images && data.images.length > 0) {
                    console.log(`[State] Found ${data.images.length} existing cropped drawings`);

                    // Pre-load full resolution images for Clean tab if not already loaded
                    for (const imgName in state.annotations) {
                        if (!state.fullResImages[imgName]) {
                            const imgElement = new Image();
                            imgElement.crossOrigin = 'anonymous';

                            await new Promise((resolve, reject) => {
                                imgElement.onload = () => {
                                    state.fullResImages[imgName] = imgElement;
                                    console.log(`Loaded full-res image for Clean tab: ${imgName}`);
                                    resolve();
                                };
                                imgElement.onerror = () => {
                                    console.error(`Failed to load image: ${imgName}`);
                                    reject();
                                };
                                imgElement.src = `/api/project/${projectId}/image/${encodeURIComponent(imgName)}`;
                            });
                        }
                    }

                    console.log('[State] Full-resolution images ready for Clean tab');
                }
            } catch (error) {
                console.error('Error loading existing crops:', error);
            }
        }

        // Load existing OCR results from project
        async function loadExistingOCRResults(projectId) {
            try {
                console.log('[OCR] Loading OCR results from project...');

                // Use the new dedicated endpoint
                const response = await fetch(`/api/project/${projectId}/ocr_results`);
                const data = await response.json();

                console.log('   OCR API response:', data);

                if (data.success && data.results && data.results.length > 0) {
                    console.log(`   Loading ${data.results.length} OCR results...`);

                    // Restore OCR results to state
                    state.ocrResults = {};
                    data.results.forEach(result => {
                        if (result.key && result.texts) {
                            state.ocrResults[result.key] = result.texts;
                            console.log(`      ${result.key}: ${result.texts.length} texts`);
                        }
                    });
                    console.log(`[OCR] Loaded ${Object.keys(state.ocrResults).length} OCR results`);
                } else {
                    console.log('[OCR] No OCR results found in project');
                }

                // Also load corrections if they exist
                console.log('[OCR] Loading OCR corrections from project...');
                const correctionsResponse = await fetch(`/api/project/${projectId}/ocr_corrections`);
                const correctionsData = await correctionsResponse.json();

                console.log('   Corrections API response:', correctionsData);

                if (correctionsData.success && correctionsData.corrections) {
                    state.corrections = correctionsData.corrections;
                    console.log(`[OCR] Loaded ${Object.keys(state.corrections).length} OCR corrections`);
                } else {
                    console.log('[OCR] No corrections found in project');
                }
            } catch (error) {
                console.error('[OCR] Error loading existing OCR results:', error);
            }
        }

        // Update project workflow status
        async function updateProjectWorkflowStatus(updates) {
            if (!currentProject) return;

            try {
                const response = await fetch(`/api/project/${currentProject.project_id}/workflow_status`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(updates)
                });

                const data = await response.json();

                if (data.success) {
                    console.log('Workflow status updated');
                }
            } catch (error) {
                console.error('Error updating workflow status:', error);
            }
        }

        // Delete project
        async function deleteProject(projectId) {
            try {
                const response = await fetch(`/api/project/${projectId}`, {
                    method: 'DELETE'
                });

                const data = await response.json();

                if (data.success) {
                    console.log('Project deleted:', projectId);
                    await loadProjects();
                } else {
                    alert('Failed to delete project: ' + data.error);
                }
            } catch (error) {
                console.error('Error deleting project:', error);
                alert('Failed to delete project');
            }
        }

        // Load project images
        // New project modal handlers
        document.getElementById('newProjectBtn').addEventListener('click', () => {
            document.getElementById('newProjectModal').classList.remove('hidden');
        });

        document.querySelectorAll('.create-project-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.getElementById('newProjectModal').classList.remove('hidden');
            });
        });

        document.getElementById('closeNewProjectBtn').addEventListener('click', () => {
            document.getElementById('newProjectModal').classList.add('hidden');
        });

        document.getElementById('cancelNewProjectBtn').addEventListener('click', () => {
            document.getElementById('newProjectModal').classList.add('hidden');
        });

        // Create new project
        document.getElementById('newProjectForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const name = document.getElementById('projectNameInput').value.trim();
            const description = document.getElementById('projectDescInput').value.trim();

            if (!name) {
                alert('Please enter a project name');
                return;
            }

            try {
                const response = await fetch('/api/project/create', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ name, description })
                });

                const data = await response.json();

                if (data.success) {
                    console.log('Project created:', data.project);

                    // Close modal and reset form BEFORE showing loading
                    document.getElementById('newProjectModal').classList.add('hidden');
                    document.getElementById('newProjectForm').reset();

                    // NOW show loading overlay (after modal is closed)
                    const overlay = document.getElementById('loadTabLoadingOverlay');
                    const loadingText = document.getElementById('loadTabLoadingText');
                    overlay.classList.remove('hidden');
                    loadingText.textContent = 'Loading projects...';

                    // Reload projects
                    await loadProjects();

                    // Optionally open the new project (will update loading text)
                    await openProject(data.project.project_id);
                } else {
                    alert('Failed to create project: ' + data.error);
                }
            } catch (error) {
                console.error('Error creating project:', error);
                document.getElementById('loadTabLoadingOverlay').classList.add('hidden');
                alert('Failed to create project');
            }
        });

        // Load projects on startup
        loadProjects();

        // Copy Citation Helper Function
        window.copyCitation = function (elementId = 'citation-text', btnElement) {
            const textEl = document.getElementById(elementId);
            if (!textEl) return;
            const text = (textEl.innerText || textEl.textContent).replace(/^"|"$/g, '').trim();
            navigator.clipboard.writeText(text).then(() => {
                const btn = btnElement || (window.event && window.event.target ? window.event.target.closest('.mac-copy-link') : null) || document.querySelector('.mac-copy-link');
                if (btn) {
                    const orig = btn.innerHTML;
                    btn.innerHTML = '<i class="bi bi-check2"></i> Copied!';
                    btn.classList.add('copied');
                    setTimeout(() => {
                        btn.innerHTML = orig;
                        btn.classList.remove('copied');
                    }, 2000);
                }
            }).catch(err => {
                console.error('Failed to copy citation:', err);
            });
        };

        // ==========================================
        // INFO MODAL
        // ==========================================
        document.getElementById('infoBtn').addEventListener('click', async () => {
            const modal = document.getElementById('infoModal');
            const modalContent = modal.querySelector('.modal-content');

            modal.classList.remove('hidden');
            // Trigger reflow to ensure animation plays
            modal.offsetHeight;
            modal.classList.add('show');

            // Fetch system info
            try {
                const response = await fetch('/api/system-info');
                const data = await response.json();
                const cpuEl = document.getElementById('systemCPU');
                const gpuEl = document.getElementById('systemGPU');

                if (cpuEl) {
                    const cores = data.cpu?.cores || navigator.hardwareConcurrency || 1;
                    const platform = data.cpu?.platform || '';
                    cpuEl.innerHTML = `<i class="bi bi-cpu me-1"></i> ${cores} Cores${platform ? ` (${platform})` : ''}`;
                }

                if (gpuEl) {
                    const cuda = (data.gpu && data.gpu.cuda_available) || data.cuda_available;
                    const gpuNames = (data.gpu && data.gpu.gpu_names) || (data.cuda_device_name ? [data.cuda_device_name] : []);
                    const mps = (data.mps && data.mps.mps_available) || data.mps_available;

                    if (cuda) {
                        const name = gpuNames.length > 0 ? gpuNames[0] : 'NVIDIA CUDA';
                        gpuEl.innerHTML = `<i class="bi bi-gpu-card me-1"></i> ${name} (CUDA)`;
                        gpuEl.className = 'chip-active';
                    } else if (mps) {
                        gpuEl.innerHTML = `<i class="bi bi-gpu-card me-1"></i> Apple Silicon (MPS)`;
                        gpuEl.className = 'chip-active';
                    } else {
                        gpuEl.innerHTML = `<i class="bi bi-gpu-card me-1"></i> CPU Only`;
                        gpuEl.className = 'chip-cpu-only';
                    }
                }
            } catch (err) {
                console.error('Error fetching system info:', err);
                const cpuEl = document.getElementById('systemCPU');
                const gpuEl = document.getElementById('systemGPU');
                if (cpuEl) cpuEl.innerHTML = '<i class="bi bi-cpu me-1"></i> Available';
                if (gpuEl) {
                    gpuEl.innerHTML = '<i class="bi bi-gpu-card me-1"></i> CPU Only';
                    gpuEl.className = 'chip-cpu-only';
                }
            }
        });

        function closeModal() {
            const modal = document.getElementById('infoModal');
            const modalContent = modal.querySelector('.modal-content');

            modalContent.classList.add('closing');
            modal.classList.remove('show');

            setTimeout(() => {
                modal.classList.add('hidden');
                modalContent.classList.remove('closing');
            }, 300);
        }

        document.getElementById('closeInfoBtn').addEventListener('click', closeModal);

        document.getElementById('infoModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('infoModal')) {
                closeModal();
            }
        });

        document.getElementById('newProjectModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('newProjectModal')) {
                document.getElementById('newProjectModal').classList.add('hidden');
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const infoModal = document.getElementById('infoModal');
                if (infoModal && !infoModal.classList.contains('hidden')) {
                    closeModal();
                }
                const newProjectModal = document.getElementById('newProjectModal');
                if (newProjectModal && !newProjectModal.classList.contains('hidden')) {
                    newProjectModal.classList.add('hidden');
                }
                const appDialogModal = document.getElementById('appDialogModal');
                if (appDialogModal && !appDialogModal.classList.contains('hidden')) {
                    closeAppDialog(currentDialogMode === 'prompt' ? null : false);
                }
                const imageZoomModal = document.getElementById('imageZoomModal');
                if (imageZoomModal && !imageZoomModal.classList.contains('hidden')) {
                    imageZoomModal.classList.add('hidden');
                }
            }
        });

        // Global state
        const state = {
            images: [],
            currentImageIndex: 0,
            annotations: {}, // { imageName: { metadata, drawings: [] } }
            ocrResults: {},  // { key: text }
            corrections: {}, // { key: correctedText }
            cleanedDrawings: {}, // { key: true/false }

            // Canvas state
            fullResImages: {}, // Store original images
            drawings: [],
            selectedDrawing: null,
            selectedTextBox: null,
            isDrawing: false,
            mode: 'idle',
            currentBox: null,
            dragAction: null,      // null, 'move', or 'resize'
            dragHandle: null,      // 'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'
            dragStartCoords: null, // { x, y }
            originalBoxCoords: null, // { x, y, w, h }

            // Scale management
            canvasScale: 1,         // Scale factor: canvas dimensions / full res dimensions
            maxCanvasWidth: 1600    // Maximum canvas width for performance
        };

        let canvas, ctx, currentImage = null, displayScale = 1;

        // Auto-save badge indicator
        function showAutosaveBadge() {
            const badge = document.getElementById('autosaveBadge');
            if (badge) {
                badge.classList.add('visible');
                clearTimeout(badge._timeout);
                badge._timeout = setTimeout(() => {
                    badge.classList.remove('visible');
                }, 1800);
            }
        }

        // Tab Management
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (btn.disabled) return;
                switchTab(btn.dataset.tab);
            });
        });

        function switchTab(tabName) {
            console.log('Switching to tab:', tabName);

            // If leaving clean tab, commit any rotation and auto-save
            const activeCleanTab = document.querySelector('.tab-content[data-tab="clean"]:not(.hidden)');
            if (activeCleanTab && tabName !== 'clean' && typeof commitCleanRotation === 'function' && typeof saveCurrentCleanDrawing === 'function') {
                commitCleanRotation();
                saveCurrentCleanDrawing();
            }

            document.querySelectorAll('.tab-btn').forEach(btn => {
                const isActive = btn.dataset.tab === tabName;
                btn.classList.toggle('active', isActive);
                btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
            });

            document.querySelectorAll('.tab-content').forEach(content => {
                const isActive = content.dataset.tab === tabName;
                content.classList.toggle('hidden', !isActive);
                content.classList.toggle('active', isActive);
            });

            // Load data when switching to specific tabs
            if (tabName === 'clean' && currentProject) {
                loadCleanTabData(currentProject.project_id);
            } else if (tabName === 'process') {
                updateOcrQueueStats();
                renderInitialOcrTerminal();
            } else if (tabName === 'annotate' && state.images.length > 0) {
                // Load annotation for current image when switching to annotate tab
                loadImageForAnnotation();
            } else if (tabName === 'review') {
                loadReviewTexts();
            } else if (tabName === 'export') {
                updateExportSummary();
            } else if (tabName === 'parser' && currentProject) {
                // Load few-shot examples from project when entering parser tab
                loadFewShotExamplesFromProject(currentProject.project_id);
            }

            console.log('Tab switched successfully to:', tabName);
        }

        function enableTab(tabName) {
            console.log('Enabling tab:', tabName);
            const btn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
            if (btn) {
                btn.disabled = false;
                console.log('Tab enabled:', tabName);
            } else {
                console.error('Tab button not found for:', tabName);
            }
        }

        // TAB 1: Load Images - File Filtering & Upload
        const VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'];

        function filterImageFiles(fileList) {
            const files = [];
            for (let i = 0; i < fileList.length; i++) {
                const f = fileList[i];
                if (!f) continue;
                const ext = '.' + f.name.split('.').pop().toLowerCase();
                if ((f.type && f.type.startsWith('image/')) || VALID_IMAGE_EXTENSIONS.includes(ext)) {
                    files.push(f);
                }
            }
            return files;
        }

        async function uploadFilesList(rawFiles) {
            if (!currentProject) {
                alert('Please select or create a project first');
                switchTab('projects');
                return;
            }

            const files = filterImageFiles(rawFiles);
            if (!files || files.length === 0) {
                alert('No supported image files found (PNG, JPEG, WebP, TIFF, BMP).');
                return;
            }

            // Show loading overlay
            const overlay = document.getElementById('loadTabLoadingOverlay');
            const loadingText = document.getElementById('loadTabLoadingText');
            overlay.classList.remove('hidden');
            loadingText.textContent = `Uploading ${files.length} image(s)...`;

            try {
                // Upload files to project
                const formData = new FormData();
                files.forEach(file => {
                    formData.append('files', file);
                });

                const uploadResponse = await fetch(`/api/project/${currentProject.project_id}/upload_images`, {
                    method: 'POST',
                    body: formData
                });

                const uploadData = await uploadResponse.json();

                if (!uploadData.success) {
                    overlay.classList.add('hidden');
                    alert('Failed to upload images: ' + (uploadData.error || 'Unknown error'));
                    return;
                }

                console.log(`Uploaded ${uploadData.uploaded} images to project`);
                loadingText.textContent = 'Loading images and generating thumbnails...';

                // Load images from project
                await loadProjectImagesForWork();

                // Hide loading overlay
                overlay.classList.add('hidden');
            } catch (err) {
                console.error('Error uploading images:', err);
                overlay.classList.add('hidden');
                alert('Error uploading images: ' + err.message);
            }
        }

        // Folder selection handler (supporting showDirectoryPicker with fallback to file input)
        const handleSelectFolder = async () => {
            if (!currentProject) {
                alert('Please select or create a project first');
                switchTab('projects');
                return;
            }

            if ('showDirectoryPicker' in window) {
                try {
                    const dirHandle = await window.showDirectoryPicker();
                    const files = [];

                    for await (const entry of dirHandle.values()) {
                        if (entry.kind === 'file') {
                            const file = await entry.getFile();
                            files.push(file);
                        }
                    }

                    await uploadFilesList(files);
                } catch (err) {
                    if (err.name !== 'AbortError') {
                        console.error('DirectoryPicker error, falling back:', err);
                        document.getElementById('folderFileInput')?.click();
                    }
                }
            } else {
                document.getElementById('folderFileInput')?.click();
            }
        };

        // File selection handler
        const handleSelectFiles = () => {
            if (!currentProject) {
                alert('Please select or create a project first');
                switchTab('projects');
                return;
            }
            document.getElementById('imageFileInput')?.click();
        };

        // Connect file inputs
        const imageFileInput = document.getElementById('imageFileInput');
        imageFileInput?.addEventListener('change', async (e) => {
            if (e.target.files && e.target.files.length > 0) {
                await uploadFilesList(Array.from(e.target.files));
                e.target.value = '';
            }
        });

        const folderFileInput = document.getElementById('folderFileInput');
        folderFileInput?.addEventListener('change', async (e) => {
            if (e.target.files && e.target.files.length > 0) {
                await uploadFilesList(Array.from(e.target.files));
                e.target.value = '';
            }
        });

        document.getElementById('selectFilesBtn')?.addEventListener('click', handleSelectFiles);
        document.getElementById('addFilesBtn')?.addEventListener('click', handleSelectFiles);
        document.getElementById('selectFolderBtn')?.addEventListener('click', handleSelectFolder);
        document.getElementById('changeFolderBtn')?.addEventListener('click', handleSelectFolder);

        // Extract files from drag and drop events (supporting folders & individual files)
        async function getFilesFromDataTransfer(dataTransfer) {
            const files = [];
            const queue = [];

            if (dataTransfer.items && dataTransfer.items.length > 0) {
                for (let i = 0; i < dataTransfer.items.length; i++) {
                    const item = dataTransfer.items[i];
                    if (item.kind === 'file') {
                        const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
                        if (entry) {
                            queue.push(entry);
                        } else {
                            const f = item.getAsFile();
                            if (f) files.push(f);
                        }
                    }
                }

                while (queue.length > 0) {
                    const entry = queue.shift();
                    if (entry.isFile) {
                        const file = await new Promise((resolve) => entry.file(resolve, () => resolve(null)));
                        if (file) files.push(file);
                    } else if (entry.isDirectory) {
                        const reader = entry.createReader();
                        const readEntries = async () => {
                            const entries = await new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
                            if (entries && entries.length > 0) {
                                for (const child of entries) {
                                    queue.push(child);
                                }
                                await readEntries();
                            }
                        };
                        await readEntries();
                    }
                }
            } else if (dataTransfer.files && dataTransfer.files.length > 0) {
                for (let i = 0; i < dataTransfer.files.length; i++) {
                    files.push(dataTransfer.files[i]);
                }
            }

            return files;
        }

        // Setup Drag & Drop on DropZone
        const dropZone = document.getElementById('dropZone');
        if (dropZone) {
            dropZone.addEventListener('click', (e) => {
                if (!e.target.closest('button')) {
                    handleSelectFiles();
                }
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.add('drag-over');
                });
            });

            ['dragleave', 'dragend'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.remove('drag-over');
                });
            });

            dropZone.addEventListener('drop', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('drag-over');

                const droppedFiles = await getFilesFromDataTransfer(e.dataTransfer);
                if (droppedFiles.length > 0) {
                    await uploadFilesList(droppedFiles);
                }
            });
        }

        // Global Drag & Drop on Load Tab Container
        const loadTabContainer = document.querySelector('.tab-content[data-tab="load"]');
        if (loadTabContainer) {
            ['dragenter', 'dragover'].forEach(eventName => {
                loadTabContainer.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (dropZone && !dropZone.classList.contains('drag-over')) {
                        dropZone.classList.add('drag-over');
                    }
                });
            });

            loadTabContainer.addEventListener('dragleave', (e) => {
                if (e.target === loadTabContainer && dropZone) {
                    dropZone.classList.remove('drag-over');
                }
            });

            loadTabContainer.addEventListener('drop', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (dropZone) dropZone.classList.remove('drag-over');

                const droppedFiles = await getFilesFromDataTransfer(e.dataTransfer);
                if (droppedFiles.length > 0) {
                    await uploadFilesList(droppedFiles);
                }
            });
        }

        // Load images from current project (when uploading new images)
        async function loadProjectImagesForWork() {
            if (!currentProject) return;

            // Use the same function as when opening a project
            await loadProjectImagesWithAnnotations(currentProject.project_id);
        }

        const handleStartAnnotation = () => {
            console.log('Start annotation clicked, images count:', state.images.length);
            if (state.images.length === 0) {
                alert('No images loaded. Please load images first.');
                return;
            }
            state.currentImageIndex = 0;
            console.log('Switching to annotate tab...');
            enableTab('annotate');
            switchTab('annotate');
            loadImageForAnnotation();
        };

        document.getElementById('startAnnotationBtn')?.addEventListener('click', handleStartAnnotation);
        document.getElementById('startAnnotationBottomBtn')?.addEventListener('click', handleStartAnnotation);

        // TAB 2: Annotation with Canvas (RESPONSIVE)
        async function loadImageForAnnotation() {
            const img = state.images[state.currentImageIndex];
            console.log('Loading image for annotation:', img);

            // Try to load annotations from project
            if (currentProject) {
                console.log('Loading annotations for:', img.name);
                const savedAnnotations = await loadAnnotationsFromProject(img.name);
                if (savedAnnotations) {
                    console.log('Found saved annotations:', savedAnnotations);
                    state.annotations[img.name] = savedAnnotations;
                } else {
                    console.log('No saved annotations found');
                }
            }

            if (!state.annotations[img.name]) {
                // Extract filename without extension for table name
                const fileNameWithoutExt = img.name.replace(/\.[^/.]+$/, '');

                state.annotations[img.name] = {
                    metadata: { tableName: fileNameWithoutExt, context: '', notes: '' },
                    drawings: []
                };
                console.log('Created new annotation structure for:', img.name);
            }

            const annotation = state.annotations[img.name];
            state.drawings = annotation.drawings;

            document.getElementById('tableName').value = annotation.metadata.tableName || '';
            document.getElementById('contextInfo').value = annotation.metadata.context || '';
            document.getElementById('notesInfo').value = annotation.metadata.notes || '';
            document.getElementById('imageIndicator').textContent = `${state.currentImageIndex + 1} / ${state.images.length}`;

            // Load FULL RES image
            console.log('Loading full res image from URL:', img.url);
            const fullResImage = new Image();
            fullResImage.onload = () => {
                console.log('Image loaded successfully, size:', fullResImage.width, 'x', fullResImage.height);
                state.fullResImages[img.name] = fullResImage;
                currentImage = fullResImage;
                initCanvas();
            };
            fullResImage.onerror = (e) => {
                console.error('Error loading image:', e);
                alert('Failed to load image: ' + img.name);
            };
            fullResImage.src = img.url;

            updateAnnotationUI();
            updateProgress();
        }

        function initCanvas() {
            if (!canvas) {
                canvas = document.getElementById('annotationCanvas');
                ctx = canvas.getContext('2d');

                canvas.addEventListener('mousedown', handleMouseDown);
                canvas.addEventListener('mousemove', handleMouseMove);
                canvas.addEventListener('mouseup', handleMouseUp);
            }

            // Calculate canvas scale for downsampling (max 1600px width)
            const fullResWidth = currentImage.width;
            const fullResHeight = currentImage.height;

            if (fullResWidth > state.maxCanvasWidth) {
                // Need to downsample
                state.canvasScale = state.maxCanvasWidth / fullResWidth;
                canvas.width = state.maxCanvasWidth;
                canvas.height = Math.round(fullResHeight * state.canvasScale);
                console.log(`Downsampling: ${fullResWidth}x${fullResHeight} → ${canvas.width}x${canvas.height} (scale: ${state.canvasScale.toFixed(3)})`);
            } else {
                // Use full resolution
                state.canvasScale = 1;
                canvas.width = fullResWidth;
                canvas.height = fullResHeight;
                console.log(`Using full resolution: ${canvas.width}x${canvas.height}`);
            }

            // Calculate display scale for responsive container
            resizeAnnotationCanvas();
        }

        function resizeAnnotationCanvas() {
            if (!canvas || !currentImage) return;
            const container = canvas.parentElement;
            if (!container) return;
            const maxWidth = container.clientWidth - 32;
            displayScale = Math.min(1, maxWidth / canvas.width);
            canvas.style.width = (canvas.width * displayScale) + 'px';
            canvas.style.height = (canvas.height * displayScale) + 'px';
            console.log(`Display scale: ${displayScale.toFixed(3)}, canvas style: ${canvas.style.width} x ${canvas.style.height}`);
            drawCanvas();
        }

        window.addEventListener('resize', () => {
            const annotateTab = document.querySelector('[data-tab="annotate"]');
            if (annotateTab && !annotateTab.classList.contains('hidden')) {
                resizeAnnotationCanvas();
            }
        });

        const HANDLE_SIZE = 8;
        const HANDLE_HIT_SIZE = 14;

        function getBoxHandles(canvasBox) {
            const { x, y, w, h } = canvasBox;
            return {
                'nw': { x: x, y: y },
                'n':  { x: x + w / 2, y: y },
                'ne': { x: x + w, y: y },
                'e':  { x: x + w, y: y + h / 2 },
                'se': { x: x + w, y: y + h },
                's':  { x: x + w / 2, y: y + h },
                'sw': { x: x, y: y + h },
                'w':  { x: x, y: y + h / 2 }
            };
        }

        function getHandleAtCoords(canvasBox, coords) {
            if (!canvasBox) return null;
            const handles = getBoxHandles(canvasBox);
            for (const [handleName, pt] of Object.entries(handles)) {
                if (Math.abs(coords.x - pt.x) <= HANDLE_HIT_SIZE / 2 &&
                    Math.abs(coords.y - pt.y) <= HANDLE_HIT_SIZE / 2) {
                    return handleName;
                }
            }
            return null;
        }

        function getCursorForHandle(handleName) {
            switch (handleName) {
                case 'nw': case 'se': return 'nwse-resize';
                case 'ne': case 'sw': return 'nesw-resize';
                case 'n': case 's': return 'ns-resize';
                case 'e': case 'w': return 'ew-resize';
                default: return 'default';
            }
        }

        function isPointInsideBox(box, pt) {
            return pt.x >= box.x && pt.x <= box.x + box.w &&
                   pt.y >= box.y && pt.y <= box.y + box.h;
        }

        function getActiveBox() {
            if (state.selectedDrawing === null || !state.drawings[state.selectedDrawing]) return null;
            const drawing = state.drawings[state.selectedDrawing];
            if (state.selectedTextBox !== null && drawing.textBoxes && drawing.textBoxes[state.selectedTextBox]) {
                return { box: drawing.textBoxes[state.selectedTextBox], type: 'text', drawingIdx: state.selectedDrawing, textBoxIdx: state.selectedTextBox };
            }
            return { box: drawing, type: 'drawing', drawingIdx: state.selectedDrawing, textBoxIdx: null };
        }

        function drawCanvas() {
            if (!currentImage) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw image scaled to canvas size
            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

            // Draw annotations (convert from full-res to canvas coords)
            state.drawings.forEach((drawing, idx) => {
                const isDrawingSelected = state.selectedDrawing === idx && state.selectedTextBox === null;

                // Convert full-res coordinates to canvas coordinates
                const canvasDrawing = fullResToCanvas(drawing);

                ctx.strokeStyle = isDrawingSelected ? '#ea580c' : '#c2410c';
                ctx.lineWidth = isDrawingSelected ? 4 : 3;
                ctx.strokeRect(canvasDrawing.x, canvasDrawing.y, canvasDrawing.w, canvasDrawing.h);

                ctx.fillStyle = isDrawingSelected ? '#ea580c' : '#c2410c';
                ctx.font = 'bold 18px sans-serif';
                ctx.fillText(`D${idx + 1}`, canvasDrawing.x, canvasDrawing.y - 8);

                drawing.textBoxes.forEach((textBox, tIdx) => {
                    const isTextSelected = state.selectedDrawing === idx && state.selectedTextBox === tIdx;
                    // Convert text box coordinates to canvas
                    const canvasTextBox = fullResToCanvas(textBox);

                    ctx.strokeStyle = isTextSelected ? '#0f766e' : '#0d9488';
                    ctx.lineWidth = isTextSelected ? 3 : 2;
                    ctx.strokeRect(canvasTextBox.x, canvasTextBox.y, canvasTextBox.w, canvasTextBox.h);

                    ctx.fillStyle = isTextSelected ? '#0f766e' : '#0d9488';
                    ctx.font = '14px sans-serif';
                    ctx.fillText(`T${tIdx + 1}`, canvasTextBox.x, canvasTextBox.y - 5);

                    drawArrow(
                        canvasTextBox.x + canvasTextBox.w / 2,
                        canvasTextBox.y + canvasTextBox.h / 2,
                        canvasDrawing.x + canvasDrawing.w / 2,
                        canvasDrawing.y + canvasDrawing.h / 2,
                        isTextSelected ? '#0f766e' : '#0d9488'
                    );
                });
            });

            // Draw active selection handles
            const active = getActiveBox();
            if (active) {
                const cBox = fullResToCanvas(active.box);
                ctx.save();
                ctx.strokeStyle = active.type === 'drawing' ? '#ea580c' : '#0d9488';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.strokeRect(cBox.x - 2, cBox.y - 2, cBox.w + 4, cBox.h + 4);
                ctx.setLineDash([]);

                const handles = getBoxHandles(cBox);
                for (const pt of Object.values(handles)) {
                    ctx.fillStyle = '#ffffff';
                    ctx.strokeStyle = active.type === 'drawing' ? '#c2410c' : '#0d9488';
                    ctx.lineWidth = 2;
                    ctx.fillRect(pt.x - HANDLE_SIZE / 2, pt.y - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
                    ctx.strokeRect(pt.x - HANDLE_SIZE / 2, pt.y - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
                }
                ctx.restore();
            }

            if (state.currentBox) {
                const color = state.mode === 'drawing' ? '#c2410c' : '#0d9488';
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(state.currentBox.x, state.currentBox.y, state.currentBox.w, state.currentBox.h);
                ctx.setLineDash([]);
            }
        }

        function drawArrow(x1, y1, x2, y2, color) {
            const headlen = 15;
            const angle = Math.atan2(y2 - y1, x2 - x1);

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 3]);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - headlen * Math.cos(angle - Math.PI / 6), y2 - headlen * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(x2 - headlen * Math.cos(angle + Math.PI / 6), y2 - headlen * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fill();
        }

        function getCanvasCoords(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        // Convert canvas coordinates to full-res image coordinates
        function canvasToFullRes(canvasCoords) {
            return {
                x: Math.round(canvasCoords.x / state.canvasScale),
                y: Math.round(canvasCoords.y / state.canvasScale),
                w: Math.round(canvasCoords.w / state.canvasScale),
                h: Math.round(canvasCoords.h / state.canvasScale)
            };
        }

        // Convert full-res image coordinates to canvas coordinates
        function fullResToCanvas(fullResCoords) {
            return {
                x: Math.round(fullResCoords.x * state.canvasScale),
                y: Math.round(fullResCoords.y * state.canvasScale),
                w: Math.round(fullResCoords.w * state.canvasScale),
                h: Math.round(fullResCoords.h * state.canvasScale)
            };
        }

        function handleMouseDown(e) {
            const coords = getCanvasCoords(e);

            if (state.mode === 'drawing' || state.mode === 'text') {
                state.isDrawing = true;
                state.currentBox = { x: coords.x, y: coords.y, w: 0, h: 0 };
                return;
            }

            // In idle mode: test handles first if an active box is selected
            const active = getActiveBox();
            if (active) {
                const cBox = fullResToCanvas(active.box);
                const handle = getHandleAtCoords(cBox, coords);
                if (handle) {
                    state.dragAction = 'resize';
                    state.dragHandle = handle;
                    state.dragStartCoords = coords;
                    state.originalBoxCoords = { ...active.box };
                    return;
                }
            }

            // Next test text boxes (top-most priority)
            for (let dIdx = 0; dIdx < state.drawings.length; dIdx++) {
                const drawing = state.drawings[dIdx];
                if (drawing.textBoxes) {
                    for (let tIdx = 0; tIdx < drawing.textBoxes.length; tIdx++) {
                        const tBox = fullResToCanvas(drawing.textBoxes[tIdx]);
                        if (isPointInsideBox(tBox, coords)) {
                            state.selectedDrawing = dIdx;
                            state.selectedTextBox = tIdx;
                            state.dragAction = 'move';
                            state.dragStartCoords = coords;
                            state.originalBoxCoords = { ...drawing.textBoxes[tIdx] };
                            drawCanvas();
                            updateAnnotationUI();
                            return;
                        }
                    }
                }
            }

            // Next test drawing boxes
            for (let dIdx = 0; dIdx < state.drawings.length; dIdx++) {
                const dBox = fullResToCanvas(state.drawings[dIdx]);
                if (isPointInsideBox(dBox, coords)) {
                    state.selectedDrawing = dIdx;
                    state.selectedTextBox = null;
                    state.dragAction = 'move';
                    state.dragStartCoords = coords;
                    state.originalBoxCoords = { ...state.drawings[dIdx] };
                    drawCanvas();
                    updateAnnotationUI();
                    return;
                }
            }

            // Clicked on empty canvas space: deselect
            state.selectedDrawing = null;
            state.selectedTextBox = null;
            state.dragAction = null;
            drawCanvas();
            updateAnnotationUI();
        }

        function handleMouseMove(e) {
            const coords = getCanvasCoords(e);

            if (state.isDrawing && state.currentBox) {
                state.currentBox.w = coords.x - state.currentBox.x;
                state.currentBox.h = coords.y - state.currentBox.y;
                drawCanvas();
                return;
            }

            const active = getActiveBox();

            if (state.dragAction === 'move' && active && state.originalBoxCoords) {
                const dxCanvas = coords.x - state.dragStartCoords.x;
                const dyCanvas = coords.y - state.dragStartCoords.y;
                const dxFull = Math.round(dxCanvas / state.canvasScale);
                const dyFull = Math.round(dyCanvas / state.canvasScale);

                const maxW = currentImage ? currentImage.width : 5000;
                const maxH = currentImage ? currentImage.height : 5000;

                active.box.x = Math.max(0, Math.min(maxW - active.box.w, state.originalBoxCoords.x + dxFull));
                active.box.y = Math.max(0, Math.min(maxH - active.box.h, state.originalBoxCoords.y + dyFull));
                drawCanvas();
                return;
            }

            if (state.dragAction === 'resize' && active && state.originalBoxCoords) {
                const dxCanvas = coords.x - state.dragStartCoords.x;
                const dyCanvas = coords.y - state.dragStartCoords.y;
                const dxFull = Math.round(dxCanvas / state.canvasScale);
                const dyFull = Math.round(dyCanvas / state.canvasScale);
                const orig = state.originalBoxCoords;
                const minDim = 15;

                let newX = orig.x, newY = orig.y, newW = orig.w, newH = orig.h;

                if (state.dragHandle.includes('e')) {
                    newW = Math.max(minDim, orig.w + dxFull);
                }
                if (state.dragHandle.includes('w')) {
                    const proposedW = orig.w - dxFull;
                    if (proposedW >= minDim) {
                        newX = orig.x + dxFull;
                        newW = proposedW;
                    }
                }
                if (state.dragHandle.includes('s')) {
                    newH = Math.max(minDim, orig.h + dyFull);
                }
                if (state.dragHandle.includes('n')) {
                    const proposedH = orig.h - dyFull;
                    if (proposedH >= minDim) {
                        newY = orig.y + dyFull;
                        newH = proposedH;
                    }
                }

                active.box.x = newX;
                active.box.y = newY;
                active.box.w = newW;
                active.box.h = newH;
                drawCanvas();
                return;
            }

            // Hover cursor feedback when in idle mode
            if (state.mode === 'idle') {
                if (active) {
                    const cBox = fullResToCanvas(active.box);
                    const handle = getHandleAtCoords(cBox, coords);
                    if (handle) {
                        canvas.style.cursor = getCursorForHandle(handle);
                        return;
                    }
                    if (isPointInsideBox(cBox, coords)) {
                        canvas.style.cursor = 'move';
                        return;
                    }
                }

                let overAny = false;
                for (const d of state.drawings) {
                    if (isPointInsideBox(fullResToCanvas(d), coords)) { overAny = true; break; }
                    if (d.textBoxes) {
                        for (const t of d.textBoxes) {
                            if (isPointInsideBox(fullResToCanvas(t), coords)) { overAny = true; break; }
                        }
                    }
                    if (overAny) break;
                }
                canvas.style.cursor = overAny ? 'pointer' : 'default';
            } else {
                canvas.style.cursor = 'crosshair';
            }
        }

        function handleMouseUp() {
            if (state.dragAction === 'move' || state.dragAction === 'resize') {
                state.dragAction = null;
                state.dragHandle = null;
                state.dragStartCoords = null;
                state.originalBoxCoords = null;
                saveCurrentAnnotations();
                showAutosaveBadge();
                updateAnnotationUI();
                updateProgress();
                return;
            }

            if (!state.currentBox || Math.abs(state.currentBox.w) < 10 || Math.abs(state.currentBox.h) < 10) {
                state.isDrawing = false;
                state.currentBox = null;
                return;
            }

            // Normalize box (handle negative width/height)
            const canvasBox = {
                x: state.currentBox.w < 0 ? state.currentBox.x + state.currentBox.w : state.currentBox.x,
                y: state.currentBox.h < 0 ? state.currentBox.y + state.currentBox.h : state.currentBox.y,
                w: Math.abs(state.currentBox.w),
                h: Math.abs(state.currentBox.h)
            };

            const fullResBox = canvasToFullRes(canvasBox);

            if (state.mode === 'drawing') {
                const newIdx = state.drawings.length;
                state.drawings.push({
                    x: fullResBox.x,
                    y: fullResBox.y,
                    w: fullResBox.w,
                    h: fullResBox.h,
                    textBoxes: []
                });
                state.selectedDrawing = newIdx;
                state.selectedTextBox = null;
                state.mode = 'idle';
                saveCurrentAnnotations();
                showAutosaveBadge();
            } else if (state.mode === 'text' && state.selectedDrawing !== null) {
                const newTIdx = state.drawings[state.selectedDrawing].textBoxes.length;
                state.drawings[state.selectedDrawing].textBoxes.push({
                    x: fullResBox.x,
                    y: fullResBox.y,
                    w: fullResBox.w,
                    h: fullResBox.h
                });
                state.selectedTextBox = newTIdx;
                state.mode = 'idle';
                saveCurrentAnnotations();
                showAutosaveBadge();
            }

            state.isDrawing = false;
            state.currentBox = null;
            drawCanvas();
            updateAnnotationUI();
            updateProgress();
        }

        function saveCurrentAnnotations() {
            const img = state.images[state.currentImageIndex];
            if (!state.annotations[img.name]) {
                state.annotations[img.name] = {
                    metadata: { tableName: '', context: '', notes: '' },
                    drawings: []
                };
            }
            state.annotations[img.name].drawings = state.drawings;
            state.annotations[img.name].metadata = {
                tableName: document.getElementById('tableName').value,
                context: document.getElementById('contextInfo').value,
                notes: document.getElementById('notesInfo').value
            };

            // Save to project backend
            if (currentProject) {
                saveAnnotationsToProject(img.name, state.annotations[img.name]);
            }
        }

        // Save annotations to project backend
        async function saveAnnotationsToProject(imageName, annotationData) {
            if (!currentProject) return;

            try {
                const response = await fetch(`/api/project/${currentProject.project_id}/annotations/${encodeURIComponent(imageName)}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ annotations: annotationData })
                });

                const data = await response.json();

                if (data.success) {
                    console.log(`Annotations saved for ${imageName}`);
                } else {
                    console.error(`Failed to save annotations:`, data.error);
                }
            } catch (error) {
                console.error(`Error saving annotations for ${imageName}:`, error);
            }
        }

        // Load annotations from project backend
        async function loadAnnotationsFromProject(imageName) {
            if (!currentProject) return null;

            try {
                const response = await fetch(`/api/project/${currentProject.project_id}/annotations/${encodeURIComponent(imageName)}`);
                const data = await response.json();

                if (data.success && data.annotations) {
                    return data.annotations;
                }

                return null;
            } catch (error) {
                console.error('Error loading annotations:', error);
                return null;
            }
        }

        function updateAnnotationUI() {
            const list = document.getElementById('drawingsList');
            const countBadge = document.getElementById('drawingsCountBadge');
            if (countBadge) countBadge.textContent = state.drawings.length;

            if (state.drawings.length === 0) {
                list.innerHTML = `
                    <div class="text-center py-6 px-2">
                        <i class="bi bi-bounding-box text-stone-300 text-2xl mb-1 block"></i>
                        <p class="text-xs text-stone-500 font-medium">No drawings on this plate</p>
                        <p class="text-[11px] text-stone-400 mt-0.5">Click <strong>Drawing (D)</strong> to mark a vessel</p>
                    </div>
                `;
            } else {
                list.innerHTML = state.drawings.map((d, idx) => {
                    const isDrawingSelected = state.selectedDrawing === idx && state.selectedTextBox === null;
                    const textCount = d.textBoxes ? d.textBoxes.length : 0;
                    return `
                        <div class="drawing-item-card ${isDrawingSelected ? 'selected-drawing' : ''}" onclick="selectDrawingFromList(${idx})">
                            <div class="drawing-card-header">
                                <div class="drawing-card-left">
                                    <span class="drawing-badge ${isDrawingSelected ? 'active' : ''}">D${idx + 1}</span>
                                    <span class="drawing-text-count ${textCount > 0 ? 'has-text' : ''}">
                                        <i class="bi bi-fonts"></i>${textCount} text${textCount === 1 ? '' : 's'}
                                    </span>
                                </div>
                                <button type="button" onclick="event.stopPropagation(); deleteDrawing(${idx})" class="drawing-delete-btn" title="Delete drawing D${idx + 1}">
                                    <i class="bi bi-trash3"></i>
                                </button>
                            </div>
                            <div class="drawing-card-footer">
                                <span class="drawing-px-label">${Math.round(d.w)}×${Math.round(d.h)} px</span>
                                <button type="button" onclick="event.stopPropagation(); addTextArea(${idx})" class="drawing-add-text-btn" title="Add text box for this drawing (T)">
                                    <i class="bi bi-plus-lg"></i> Add Text <span class="shortcut-hint">(T)</span>
                                </button>
                            </div>
                        </div>
                    `;
                }).join('');

            }

            const delBtn = document.getElementById('deleteSelectedBtn');
            if (delBtn) {
                if (state.selectedDrawing !== null) {
                    delBtn.classList.remove('hidden');
                    if (state.selectedTextBox !== null) {
                        delBtn.innerHTML = '<i class="bi bi-trash3 mr-1"></i> Delete Text <span class="text-xs opacity-75">(Del)</span>';
                    } else {
                        delBtn.innerHTML = `<i class="bi bi-trash3 mr-1"></i> Delete D${state.selectedDrawing + 1} <span class="text-xs opacity-75">(Del)</span>`;
                    }
                } else {
                    delBtn.classList.add('hidden');
                }
            }

            // Mode indicator in toolbar (replaces invasive statusBar banner)
            const newDrawingBtn = document.getElementById('newDrawingBtn');
            const modeBadge = document.getElementById('modeBadge');

            if (state.mode === 'drawing') {
                if (newDrawingBtn) {
                    newDrawingBtn.className = 'btn btn-primary px-3 py-1.5 text-xs font-semibold';
                }
                if (modeBadge) {
                    modeBadge.className = 'text-xs font-medium px-2.5 py-1 rounded-full bg-orange-50 text-orange-700 border border-orange-200 inline-flex items-center gap-1.5';
                    modeBadge.innerHTML = '<i class="bi bi-bounding-box text-orange-600"></i> Drawing vessel <span class="text-stone-400 font-mono text-[10px]">(Esc)</span>';
                }
            } else if (state.mode === 'text') {
                if (newDrawingBtn) {
                    newDrawingBtn.className = 'btn btn-secondary px-3 py-1.5 text-xs font-semibold';
                }
                if (modeBadge) {
                    modeBadge.className = 'text-xs font-medium px-2.5 py-1 rounded-full bg-teal-50 text-teal-700 border border-teal-200 inline-flex items-center gap-1.5';
                    modeBadge.innerHTML = `<i class="bi bi-fonts text-teal-600"></i> Text box for D${state.selectedDrawing + 1} <span class="text-stone-400 font-mono text-[10px]">(Esc)</span>`;
                }
            } else {
                if (newDrawingBtn) {
                    newDrawingBtn.className = 'btn btn-secondary px-3 py-1.5 text-xs font-semibold';
                }
                if (modeBadge) {
                    modeBadge.className = 'hidden';
                    modeBadge.innerHTML = '';
                }
            }
        }

        window.selectDrawingFromList = function(idx) {
            state.selectedDrawing = idx;
            state.selectedTextBox = null;
            drawCanvas();
            updateAnnotationUI();
        };

        function updateProgress() {
            const completed = Object.keys(state.annotations).filter(name => {
                const ann = state.annotations[name];
                return ann && ann.drawings && ann.drawings.length > 0 && ann.metadata && ann.metadata.tableName;
            }).length;

            const total = state.images.length;
            const percent = total > 0 ? Math.round((completed / total) * 100) : 0;

            const annBar = document.getElementById('annotationProgressBar');
            if (annBar) annBar.style.width = percent + '%';

            const annPct = document.getElementById('annotationPercent');
            if (annPct) annPct.textContent = percent + '%';

            const annProg = document.getElementById('annotationProgress');
            if (annProg) {
                annProg.textContent = `${completed} / ${total} plates completed`;
            }

            const annotatedSummaryEl = document.getElementById('annotatedSummaryText');
            if (annotatedSummaryEl) {
                annotatedSummaryEl.textContent = `${completed} of ${total} annotated`;
            }

            const catalogStatusEl = document.getElementById('catalogStatusText');
            if (catalogStatusEl) {
                catalogStatusEl.textContent = completed === total
                    ? 'All plates annotated'
                    : `${total - completed} plate${(total - completed) === 1 ? '' : 's'} awaiting annotation`;
            }

            const thumbnails = document.getElementById('imagesList')?.children;
            if (thumbnails) {
                state.images.forEach((img, idx) => {
                    const ann = state.annotations[img.name];
                    if (ann && ann.drawings && ann.drawings.length > 0 && ann.metadata && ann.metadata.tableName) {
                        thumbnails[idx]?.classList.add('annotated');
                    } else {
                        thumbnails[idx]?.classList.remove('annotated');
                    }
                    if (idx === state.currentImageIndex) {
                        thumbnails[idx]?.classList.add('active');
                    } else {
                        thumbnails[idx]?.classList.remove('active');
                    }
                });
            }
        }

        window.addTextArea = function (drawingIdx) {
            state.selectedDrawing = drawingIdx;
            state.selectedTextBox = null;
            state.mode = 'text';
            drawCanvas();
            updateAnnotationUI();
        };

        window.deleteDrawing = function (drawingIdx) {
            state.drawings.splice(drawingIdx, 1);
            if (state.selectedDrawing === drawingIdx) {
                state.selectedDrawing = null;
                state.selectedTextBox = null;
                state.mode = 'idle';
            }
            saveCurrentAnnotations();
            showAutosaveBadge();
            drawCanvas();
            updateAnnotationUI();
            updateProgress();
        };

        // Navigation with automatic metadata and annotation save
        document.getElementById('prevImageBtn').addEventListener('click', () => {
            if (state.currentImageIndex > 0) {
                saveCurrentAnnotations();
                showAutosaveBadge();
                state.currentImageIndex--;
                state.mode = 'idle';
                state.selectedDrawing = null;
                state.selectedTextBox = null;
                loadImageForAnnotation();
            }
        });

        document.getElementById('nextImageBtn').addEventListener('click', () => {
            if (state.currentImageIndex < state.images.length - 1) {
                saveCurrentAnnotations();
                showAutosaveBadge();
                state.currentImageIndex++;
                state.mode = 'idle';
                state.selectedDrawing = null;
                state.selectedTextBox = null;
                loadImageForAnnotation();
            }
        });

        document.getElementById('newDrawingBtn').addEventListener('click', () => {
            state.mode = 'drawing';
            state.selectedDrawing = null;
            state.selectedTextBox = null;
            updateAnnotationUI();
        });

        // Delete active box button
        document.getElementById('deleteSelectedBtn')?.addEventListener('click', () => {
            if (state.selectedDrawing === null) return;
            if (state.selectedTextBox !== null && state.drawings[state.selectedDrawing]) {
                state.drawings[state.selectedDrawing].textBoxes.splice(state.selectedTextBox, 1);
                state.selectedTextBox = null;
            } else {
                state.drawings.splice(state.selectedDrawing, 1);
                state.selectedDrawing = null;
                state.selectedTextBox = null;
            }
            saveCurrentAnnotations();
            showAutosaveBadge();
            drawCanvas();
            updateAnnotationUI();
            updateProgress();
        });

        document.getElementById('clearCurrentBtn').addEventListener('click', async () => {
            const confirmed = await showConfirmDialog({
                title: 'Clear All Drawings',
                message: 'Clear all detected and drawn vessel regions on this image? This will discard unannotated outlines.',
                confirmText: 'Clear Drawings',
                cancelText: 'Cancel',
                type: 'danger',
                icon: 'bi-trash3-fill'
            });

            if (confirmed) {
                state.drawings = [];
                state.mode = 'idle';
                state.selectedDrawing = null;
                state.selectedTextBox = null;
                saveCurrentAnnotations();
                showAutosaveBadge();
                drawCanvas();
                updateAnnotationUI();
                updateProgress();
            }
        });

        // Debounced live auto-save on metadata inputs
        let metadataDebounceTimer = null;
        ['tableName', 'contextInfo', 'notesInfo'].forEach(id => {
            document.getElementById(id)?.addEventListener('input', () => {
                clearTimeout(metadataDebounceTimer);
                metadataDebounceTimer = setTimeout(() => {
                    saveCurrentAnnotations();
                    showAutosaveBadge();
                    updateProgress();
                }, 350);
            });
        });

        document.getElementById('saveMetadataBtn').addEventListener('click', () => {
            saveCurrentAnnotations();
            showAutosaveBadge();
            updateProgress();
        });

        // Keyboard shortcuts for annotation, nudging, and deletion
        document.addEventListener('keydown', (e) => {
            const activeTab = document.querySelector('.tab-content:not(.hidden)');
            if (!activeTab || activeTab.dataset.tab !== 'annotate') return;

            // Don't trigger if user is typing in form inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            // Arrow keys - Nudge selected bounding box
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
                const active = getActiveBox();
                if (active) {
                    e.preventDefault();
                    const stepFull = Math.round((e.shiftKey ? 10 : 2) / state.canvasScale);
                    const maxW = currentImage ? currentImage.width : 5000;
                    const maxH = currentImage ? currentImage.height : 5000;

                    if (e.key === 'ArrowUp') active.box.y = Math.max(0, active.box.y - stepFull);
                    if (e.key === 'ArrowDown') active.box.y = Math.min(maxH - active.box.h, active.box.y + stepFull);
                    if (e.key === 'ArrowLeft') active.box.x = Math.max(0, active.box.x - stepFull);
                    if (e.key === 'ArrowRight') active.box.x = Math.min(maxW - active.box.w, active.box.x + stepFull);

                    saveCurrentAnnotations();
                    showAutosaveBadge();
                    drawCanvas();
                    return;
                }
            }

            // Delete or Backspace key - Delete selected box
            if (e.key === 'Delete' || e.key === 'Backspace') {
                if (state.selectedDrawing !== null) {
                    e.preventDefault();
                    if (state.selectedTextBox !== null && state.drawings[state.selectedDrawing]) {
                        state.drawings[state.selectedDrawing].textBoxes.splice(state.selectedTextBox, 1);
                        state.selectedTextBox = null;
                    } else {
                        state.drawings.splice(state.selectedDrawing, 1);
                        state.selectedDrawing = null;
                        state.selectedTextBox = null;
                    }
                    saveCurrentAnnotations();
                    showAutosaveBadge();
                    drawCanvas();
                    updateAnnotationUI();
                    updateProgress();
                    return;
                }
            }

            // Escape key - Cancel active selection or drawing mode
            if (e.key === 'Escape') {
                state.mode = 'idle';
                state.selectedDrawing = null;
                state.selectedTextBox = null;
                state.currentBox = null;
                state.isDrawing = false;
                drawCanvas();
                updateAnnotationUI();
                return;
            }

            // D key - Add new drawing
            if (e.key === 'd' || e.key === 'D') {
                e.preventDefault();
                state.mode = 'drawing';
                state.selectedDrawing = null;
                state.selectedTextBox = null;
                updateAnnotationUI();
                return;
            }

            // T key - Add text area (only if a drawing is selected)
            if (e.key === 't' || e.key === 'T') {
                e.preventDefault();
                if (state.selectedDrawing !== null) {
                    state.selectedTextBox = null;
                    state.mode = 'text';
                    updateAnnotationUI();
                } else {
                    const modeBadge = document.getElementById('modeBadge');
                    if (modeBadge) {
                        modeBadge.className = 'text-xs font-medium px-2.5 py-1 rounded-full bg-amber-50 text-amber-800 border border-amber-200 inline-flex items-center gap-1.5';
                        modeBadge.innerHTML = '<i class="bi bi-exclamation-circle text-amber-600"></i> Select a vessel drawing (D) first';
                        setTimeout(() => {
                            if (state.mode !== 'text' && state.mode !== 'drawing') {
                                modeBadge.className = 'hidden';
                                modeBadge.innerHTML = '';
                            }
                        }, 2500);
                    }
                }
                return;
            }
        });

        document.getElementById('finishAnnotationBtn').addEventListener('click', async () => {
            const hasAnnotations = Object.values(state.annotations).some(a => a.drawings.length > 0);
            if (!hasAnnotations) {
                alert('Please annotate at least one image before proceeding');
                return;
            }

            // Show loading overlay
            const overlay = document.getElementById('cropGenerationOverlay');
            const progressBar = document.getElementById('cropGenerationProgressBar');
            const progressText = document.getElementById('cropGenerationProgressText');
            overlay.classList.remove('hidden');

            try {
                // Save current annotations before switching tabs
                saveCurrentAnnotations();

                // Generate and save cropped images to project
                if (currentProject) {
                    await generateAndSaveCroppedImages(progressBar, progressText);
                }

                // Hide overlay
                overlay.classList.add('hidden');

                enableTab('process');
                switchTab('process');
            } catch (error) {
                console.error('Error processing annotations:', error);
                overlay.classList.add('hidden');
                alert('Error processing annotations: ' + error.message);
            }
        });

        // Generate and save cropped images to project
        async function generateAndSaveCroppedImages(progressBar, progressText) {
            console.log('Generating cropped images...');

            let totalCrops = 0;
            let processedCrops = 0;

            // Count total crops first
            for (const imgName in state.annotations) {
                const ann = state.annotations[imgName];
                totalCrops += ann.drawings.length; // Each drawing is a crop
            }

            if (progressText) {
                progressText.textContent = `0 / ${totalCrops} crops`;
            }

            for (const imgName in state.annotations) {
                const ann = state.annotations[imgName];
                const img = state.images.find(i => i.name === imgName);
                if (!img) continue;

                // Load full res image if not already loaded
                let imgElement = state.fullResImages[imgName];
                if (!imgElement) {
                    console.log(`Loading full-resolution image for cropping: ${imgName}`);

                    // Update overlay text
                    if (progressText) {
                        const cropText = document.getElementById('cropGenerationText');
                        if (cropText) {
                            cropText.textContent = `Loading image: ${imgName}...`;
                        }
                    }

                    // Create and load image
                    imgElement = new Image();
                    const imageLoaded = new Promise((resolve, reject) => {
                        imgElement.onload = resolve;
                        imgElement.onerror = reject;
                    });
                    imgElement.src = img.url;

                    try {
                        await imageLoaded;
                        state.fullResImages[imgName] = imgElement;
                        console.log(`Image loaded for cropping (${imgElement.width}x${imgElement.height}px)`);
                    } catch (err) {
                        console.error(`Failed to load image for cropping: ${imgName}`, err);
                        continue;
                    }

                    // Reset overlay text
                    if (progressText) {
                        const cropText = document.getElementById('cropGenerationText');
                        if (cropText) {
                            cropText.textContent = 'Generating cropped images...';
                        }
                    }
                }

                for (let dIdx = 0; dIdx < ann.drawings.length; dIdx++) {
                    const drawing = ann.drawings[dIdx];

                    // Create crop of the drawing (without text)
                    const drawingCanvas = document.createElement('canvas');
                    const drawingCtx = drawingCanvas.getContext('2d');
                    drawingCanvas.width = drawing.w;
                    drawingCanvas.height = drawing.h;

                    // Fill white background
                    drawingCtx.fillStyle = 'white';
                    drawingCtx.fillRect(0, 0, drawing.w, drawing.h);

                    // Draw the cropped area
                    drawingCtx.drawImage(imgElement, drawing.x, drawing.y, drawing.w, drawing.h, 0, 0, drawing.w, drawing.h);

                    // Save drawing crop
                    const drawingImageData = drawingCanvas.toDataURL('image/png');
                    const baseImgName = imgName.replace(/\.[^/.]+$/, '');
                    const drawingFilename = `${baseImgName}_d${dIdx + 1}.png`;

                    await saveCroppedToProject(drawingImageData, drawingFilename);

                    processedCrops++;

                    // Update progress
                    if (progressBar && progressText) {
                        const percent = Math.round((processedCrops / totalCrops) * 100);
                        progressBar.style.width = percent + '%';
                        progressText.textContent = `${processedCrops} / ${totalCrops} crops`;
                    }

                    // Create crops for each text box
                    for (let tIdx = 0; tIdx < drawing.textBoxes.length; tIdx++) {
                        const textBox = drawing.textBoxes[tIdx];

                        const textCanvas = document.createElement('canvas');
                        const textCtx = textCanvas.getContext('2d');
                        textCanvas.width = textBox.w;
                        textCanvas.height = textBox.h;

                        textCtx.drawImage(imgElement, textBox.x, textBox.y, textBox.w, textBox.h, 0, 0, textBox.w, textBox.h);

                        const textImageData = textCanvas.toDataURL('image/png');
                        const textFilename = `${baseImgName}_d${dIdx + 1}_t${tIdx + 1}.png`;

                        await saveCroppedToProject(textImageData, textFilename);
                        totalCrops++;
                    }
                }
            }

            console.log(`Saved ${processedCrops} cropped images to project`);
        }

        // Save cropped image to project
        async function saveCroppedToProject(imageData, filename) {
            if (!currentProject) return;

            try {
                const response = await fetch(`/api/project/${currentProject.project_id}/save_cropped`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image: imageData,
                        filename: filename
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    console.error('Failed to save cropped image:', filename);
                }
            } catch (error) {
                console.error('Error saving cropped image:', error);
            }
        }

        // TAB 3: Process OCR - Elegant Centered Card & Status
        function appendProgressiveResult(key, tagNum, text, isError = false) {
            const listEl = document.getElementById('ocrTranscribedList');
            const countEl = document.getElementById('ocrResultsBadgeCount');
            const summaryCard = document.getElementById('ocrResultsSummary');
            if (summaryCard) summaryCard.classList.remove('hidden');
            if (!listEl) return;

            const row = document.createElement('div');
            row.className = 'ocr-result-item';
            const safeText = String(text || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
            const textStyle = isError ? 'color: #dc2626; font-style: italic;' : 'color: #1c1917;';
            row.innerHTML = `
                <span class="font-semibold flex-shrink-0 font-mono text-[11px] px-1.5 py-0.5 rounded bg-stone-100 border border-stone-200" style="color: var(--primary);">${key} · T${tagNum}</span>
                <span class="text-right flex-1 select-all break-words text-xs font-mono" style="${textStyle}">&ldquo;${safeText}&rdquo;</span>
            `;
            listEl.appendChild(row);
            listEl.scrollTop = listEl.scrollHeight;

            if (countEl) {
                const count = listEl.children.length;
                countEl.textContent = `${count} ${count === 1 ? 'box' : 'boxes'}`;
            }
        }

        function renderOcrTranscribedList() {
            const listEl = document.getElementById('ocrTranscribedList');
            const countEl = document.getElementById('ocrResultsBadgeCount');
            if (!listEl) return 0;
            listEl.innerHTML = '';

            let count = 0;
            for (const key in state.ocrResults) {
                const texts = state.ocrResults[key];
                if (Array.isArray(texts)) {
                    texts.forEach((txt, idx) => {
                        count++;
                        const isError = (txt === 'Error');
                        const row = document.createElement('div');
                        row.className = 'ocr-result-item';
                        const safeText = String(txt || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
                        const textStyle = isError ? 'color: #dc2626; font-style: italic;' : 'color: #1c1917;';
                        row.innerHTML = `
                            <span class="font-semibold flex-shrink-0 font-mono text-[11px] px-1.5 py-0.5 rounded bg-stone-100 border border-stone-200" style="color: var(--primary);">${key} · T${idx + 1}</span>
                            <span class="text-right flex-1 select-all break-words text-xs font-mono" style="${textStyle}">&ldquo;${safeText}&rdquo;</span>
                        `;
                        listEl.appendChild(row);
                    });
                }
            }

            if (countEl) countEl.textContent = `${count} ${count === 1 ? 'box' : 'boxes'}`;
            return count;
        }

        function updateOcrQueueStats() {
            let totalBoxes = 0;
            for (const imgName in state.annotations) {
                const ann = state.annotations[imgName];
                if (ann && ann.drawings && ann.drawings.length > 0) {
                    ann.drawings.forEach(d => {
                        if (d.textBoxes) totalBoxes += d.textBoxes.length;
                    });
                }
            }

            // Count how many text boxes are already transcribed
            let processedBoxes = 0;
            if (state.ocrResults) {
                for (const key in state.ocrResults) {
                    const texts = state.ocrResults[key];
                    if (Array.isArray(texts)) {
                        processedBoxes += texts.length;
                    }
                }
            }

            const countEl = document.getElementById('ocrProgressCount');
            const startBtn = document.getElementById('startOcrBtn');
            const skipBtn = document.getElementById('skipOcrBtn');
            const proceedCleanBtn = document.getElementById('proceedCleanBtn');
            const progressBar = document.getElementById('ocrProgressBar');
            const progressPercent = document.getElementById('ocrProgressPercent');
            const progressText = document.getElementById('ocrProgressText');

            const visual = document.getElementById('ocrCenterVisual');
            const headline = document.getElementById('ocrCenterHeadline');
            const subtitle = document.getElementById('ocrCenterSubtitle');
            const liveStepPill = document.getElementById('ocrLiveStepPill');
            const resultsSummary = document.getElementById('ocrResultsSummary');
            const actionsRow = document.getElementById('ocrActionsRow');

            const isFullyProcessed = (totalBoxes > 0 && processedBoxes >= totalBoxes);

            if (countEl) {
                if (isFullyProcessed) {
                    countEl.textContent = `${totalBoxes} / ${totalBoxes} text boxes`;
                } else {
                    countEl.textContent = `${processedBoxes} / ${totalBoxes} text boxes`;
                }
            }

            if (isFullyProcessed) {
                if (startBtn && !startBtn.dataset.running) {
                    startBtn.disabled = false;
                    startBtn.innerHTML = '<i class="bi bi-arrow-repeat text-base"></i> Re-run OCR Processing';
                }
                if (skipBtn) {
                    skipBtn.classList.add('hidden');
                }
                if (proceedCleanBtn && (!startBtn || !startBtn.dataset.running)) {
                    proceedCleanBtn.classList.remove('hidden');
                }
                if (progressBar) {
                    progressBar.style.width = '100%';
                    progressBar.textContent = '100%';
                }
                if (progressPercent) progressPercent.textContent = '100%';
                if (progressText) {
                    progressText.innerHTML = '<i class="bi bi-check-circle-fill mr-1" style="color: var(--teal);"></i> Extraction complete!';
                }

                if (!startBtn || !startBtn.dataset.running) {
                    if (visual) {
                        visual.innerHTML = '<div class="w-14 h-14 rounded-full flex items-center justify-center" style="background: rgba(13, 148, 136, 0.1); color: var(--teal);"><i class="bi bi-check-lg text-3xl"></i></div>';
                    }
                    if (headline) headline.textContent = 'Text Extraction Complete!';
                    if (subtitle) subtitle.textContent = `All ${totalBoxes} text boxes transcribed and cataloged.`;
                    if (liveStepPill) liveStepPill.classList.add('hidden');
                    renderOcrTranscribedList();
                    if (resultsSummary) resultsSummary.classList.remove('hidden');
                    if (actionsRow) actionsRow.classList.remove('hidden');
                }
            } else {
                if (startBtn && !startBtn.dataset.running) {
                    startBtn.disabled = false;
                    startBtn.innerHTML = '<i class="bi bi-play-fill text-base"></i> Start OCR Processing';
                }
                if (skipBtn && !startBtn?.dataset.running) {
                    if (processedBoxes > 0) {
                        skipBtn.classList.add('hidden');
                    } else {
                        skipBtn.classList.remove('hidden');
                        skipBtn.disabled = false;
                    }
                }
                if (!startBtn || !startBtn.dataset.running) {
                    if (visual) {
                        visual.innerHTML = '<div class="w-14 h-14 rounded-full flex items-center justify-center" style="background: var(--primary-soft); color: var(--primary);"><i class="bi bi-fonts text-2xl"></i></div>';
                    }
                    if (headline) headline.textContent = 'Ready for Text Extraction';
                    if (subtitle) subtitle.textContent = 'Press "Start OCR Processing" to analyze drawing labels with the local AI vision model.';
                    if (liveStepPill) liveStepPill.classList.add('hidden');
                    if (processedBoxes > 0) {
                        renderOcrTranscribedList();
                        if (resultsSummary) resultsSummary.classList.remove('hidden');
                        if (actionsRow) actionsRow.classList.remove('hidden');
                    } else {
                        if (resultsSummary) resultsSummary.classList.add('hidden');
                        if (actionsRow) actionsRow.classList.add('hidden');
                    }
                }
            }
        }

        function renderInitialOcrTerminal() {
            const log = document.getElementById('ocrLog');
            if (!log || log.dataset.hasRun === 'true') return;
            const model = window.ocrModelName || 'GLM-OCR';
            const cuda = window.cudaAvailable ? 'CUDA Hardware Acceleration' : (navigator.platform && navigator.platform.includes('Mac') ? 'Apple Silicon (MPS)' : 'CPU Mode');
            const timeStr = new Date().toTimeString().split(' ')[0];

            log.innerHTML = `
<div class="text-stone-400 text-left leading-relaxed">[${timeStr}] Engine: ${model} (${cuda})</div>
<div class="text-stone-400 text-left leading-relaxed">[${timeStr}] Pipeline: Crop Bounding Boxes &rarr; Local VLM &rarr; Typo Normalization</div>
<div class="text-teal-400 text-left leading-relaxed">[${timeStr}] Ready for batch extraction.</div>`;
        }

        function appendOcrTerminalLine(tag, text, type = 'info') {
            const log = document.getElementById('ocrLog');
            if (!log) return;

            const timeStr = new Date().toTimeString().split(' ')[0];
            const line = document.createElement('div');
            let color = 'text-stone-300';
            if (type === 'ok' || tag.includes('OK')) color = 'text-teal-300';
            else if (type === 'err' || tag.includes('ERR')) color = 'text-rose-300';
            else if (type === 'plate' || tag.includes('PLATE')) color = 'text-amber-300';

            const safeText = String(text || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
            line.className = `${color} break-words text-left leading-relaxed py-0.5`;
            line.innerHTML = `<span class="text-stone-500 font-mono">[${timeStr}]</span> <span class="font-semibold text-stone-400 font-mono">[${tag}]</span> <span>${safeText}</span>`;
            log.appendChild(line);
            log.scrollTop = log.scrollHeight;
        }

        // Copy OCR extracted texts or log output
        document.getElementById('copyOcrResultsBtn')?.addEventListener('click', async () => {
            const btn = document.getElementById('copyOcrResultsBtn');
            if (!btn) return;

            let textToCopy = '';
            let hasResults = false;

            for (const key in state.ocrResults) {
                const texts = state.ocrResults[key];
                if (Array.isArray(texts) && texts.length > 0) {
                    hasResults = true;
                    textToCopy += `[${key}]\n`;
                    texts.forEach((txt, idx) => {
                        textToCopy += `  T${idx + 1}: ${txt}\n`;
                    });
                }
            }

            if (!hasResults) {
                const log = document.getElementById('ocrLog');
                textToCopy = log ? log.innerText : 'No OCR results available.';
            }

            try {
                await navigator.clipboard.writeText(textToCopy);
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '<i class="bi bi-check2 text-xs" style="color: var(--teal);"></i> <span>Copied to Clipboard!</span>';
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                }, 2000);
            } catch (err) {
                console.error('Failed to copy OCR results:', err);
            }
        });

        // Toggle collapsible diagnostic log with smooth CSS animation
        document.getElementById('toggleOcrDetailsBtn')?.addEventListener('click', () => {
            const details = document.getElementById('ocrCollapsibleDetails');
            const btn = document.getElementById('toggleOcrDetailsBtn');
            if (!details || !btn) return;
            const isOpen = details.classList.toggle('is-open');
            btn.innerHTML = isOpen 
                ? '<i class="bi bi-chevron-up"></i> <span>Hide Details</span>' 
                : '<i class="bi bi-chevron-down"></i> <span>Details</span>';
        });

        document.getElementById('startOcrBtn').addEventListener('click', async () => {
            const startBtn = document.getElementById('startOcrBtn');
            const skipBtn = document.getElementById('skipOcrBtn');
            const proceedCleanBtn = document.getElementById('proceedCleanBtn');
            const visual = document.getElementById('ocrCenterVisual');
            const headline = document.getElementById('ocrCenterHeadline');
            const subtitle = document.getElementById('ocrCenterSubtitle');
            const liveStepPill = document.getElementById('ocrLiveStepPill');
            const liveStepText = document.getElementById('ocrLiveStepText');
            const resultsSummary = document.getElementById('ocrResultsSummary');
            const actionsRow = document.getElementById('ocrActionsRow');

            if (startBtn.dataset.running === 'true') return;

            if (window.ocrAvailable === false) {
                alert('No OCR model is installed on this system. Use "Skip OCR (Manual Input)" to enter text manually instead.');
                return;
            }

            let totalTexts = 0;
            let imagesWithTextBoxes = 0;

            for (const imgName in state.annotations) {
                const ann = state.annotations[imgName];
                let textBoxesInImage = 0;
                if (ann && ann.drawings) {
                    ann.drawings.forEach(d => {
                        if (d.textBoxes) textBoxesInImage += d.textBoxes.length;
                    });
                }
                totalTexts += textBoxesInImage;
                if (textBoxesInImage > 0) {
                    imagesWithTextBoxes++;
                }
            }

            if (totalTexts === 0) {
                alert('No text areas to process! Please draw at least one text bounding box on a vessel in Tab 2.');
                return;
            }

            const log = document.getElementById('ocrLog');
            const progressBar = document.getElementById('ocrProgressBar');
            const progressText = document.getElementById('ocrProgressText');
            const progressPercent = document.getElementById('ocrProgressPercent');
            const countEl = document.getElementById('ocrProgressCount');

            // Reset details drawer to closed state
            const details = document.getElementById('ocrCollapsibleDetails');
            if (details) details.classList.remove('is-open');
            const toggleDetailsBtn = document.getElementById('toggleOcrDetailsBtn');
            if (toggleDetailsBtn) toggleDetailsBtn.innerHTML = '<i class="bi bi-chevron-down"></i> <span>Details</span>';

            // Set running state
            startBtn.dataset.running = 'true';
            startBtn.disabled = true;
            startBtn.innerHTML = '<i class="bi bi-arrow-repeat spin text-base"></i> Processing OCR...';
            if (skipBtn) skipBtn.classList.add('hidden');
            if (proceedCleanBtn) proceedCleanBtn.classList.add('hidden');

            // Set centered spinning wheel and active status
            if (visual) {
                visual.innerHTML = '<div class="ocr-spinner"></div>';
            }
            if (headline) headline.textContent = 'Transcribing Drawing Labels...';
            if (subtitle) subtitle.textContent = `Analyzing ${totalTexts} text boxes across ${imagesWithTextBoxes} plates with local Vision-Language Model...`;
            if (liveStepPill) liveStepPill.classList.remove('hidden');
            if (liveStepText) liveStepText.textContent = 'Initializing VLM inference pipeline...';
            
            // Clear and reveal progressive transcribed list
            const listEl = document.getElementById('ocrTranscribedList');
            if (listEl) listEl.innerHTML = '';
            const badgeCount = document.getElementById('ocrResultsBadgeCount');
            if (badgeCount) badgeCount.textContent = '0 boxes';
            if (resultsSummary) resultsSummary.classList.remove('hidden');
            if (actionsRow) actionsRow.classList.add('hidden');

            // Reset progress bar
            if (progressBar) {
                progressBar.style.width = '0%';
                progressBar.textContent = '';
            }
            if (progressPercent) progressPercent.textContent = '0%';
            if (countEl) countEl.textContent = `0 / ${totalTexts} text boxes`;
            if (progressText) {
                progressText.innerHTML = '<i class="bi bi-arrow-repeat spin mr-1"></i> Starting extraction...';
            }

            if (log) {
                log.dataset.hasRun = 'true';
                log.innerHTML = '';
            }

            let processedTexts = 0;
            let success = false;

            try {
                appendOcrTerminalLine('INIT', `Starting batch OCR extraction across ${imagesWithTextBoxes} plates (${totalTexts} text boxes)...`, 'info');

                for (const imgName in state.annotations) {
                    const ann = state.annotations[imgName];
                    const img = state.images.find(i => i.name === imgName);
                    if (!img || !ann || !ann.drawings) continue;

                    const totalTextBoxesInImage = ann.drawings.reduce((sum, d) => sum + (d.textBoxes ? d.textBoxes.length : 0), 0);
                    if (totalTextBoxesInImage === 0) continue;

                    let imgElement = state.fullResImages[imgName];
                    if (!imgElement) {
                        appendOcrTerminalLine('IMG', `Loading full-resolution image: ${imgName}...`, 'plate');
                        imgElement = new Image();
                        const imageLoaded = new Promise((resolve, reject) => {
                            imgElement.onload = resolve;
                            imgElement.onerror = reject;
                        });
                        imgElement.src = img.url;

                        try {
                            await imageLoaded;
                            state.fullResImages[imgName] = imgElement;
                            appendOcrTerminalLine('IMG OK', `${imgName} loaded (${imgElement.width}×${imgElement.height}px)`, 'ok');
                        } catch (err) {
                            appendOcrTerminalLine('ERROR', `Failed to load image ${imgName}: ${err.message}`, 'err');
                            continue;
                        }
                    } else {
                        appendOcrTerminalLine('PLATE', `${imgName} (${imgElement.width}×${imgElement.height}px)`, 'plate');
                    }

                    for (let dIdx = 0; dIdx < ann.drawings.length; dIdx++) {
                        const drawing = ann.drawings[dIdx];
                        if (!drawing.textBoxes || drawing.textBoxes.length === 0) continue;

                        const key = `${imgName}_d${dIdx + 1}`;
                        appendOcrTerminalLine('DRAW', `Drawing D${dIdx + 1} (${drawing.textBoxes.length} text areas)`, 'draw');

                        const recognizedTexts = [];

                        for (let tIdx = 0; tIdx < drawing.textBoxes.length; tIdx++) {
                            const textBox = drawing.textBoxes[tIdx];

                            if (liveStepText) {
                                liveStepText.textContent = `Drawing D${dIdx + 1} · Box T${tIdx + 1} (${Math.round(textBox.w)}×${Math.round(textBox.h)}px) — Inference...`;
                            }
                            if (subtitle) {
                                subtitle.textContent = `Transcribing box ${processedTexts + 1} of ${totalTexts} (${imgName})...`;
                            }

                            appendOcrTerminalLine('CROP', `T${tIdx + 1} bounding box (${Math.round(textBox.w)}×${Math.round(textBox.h)}px)...`, 'box');

                            const tempCanvas = document.createElement('canvas');
                            const tempCtx = tempCanvas.getContext('2d');
                            tempCanvas.width = textBox.w;
                            tempCanvas.height = textBox.h;
                            tempCtx.drawImage(imgElement, textBox.x, textBox.y, textBox.w, textBox.h, 0, 0, textBox.w, textBox.h);

                            const textImageData = tempCanvas.toDataURL('image/png');

                            try {
                                const response = await fetch('/api/ocr', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ image: textImageData })
                                });

                                const result = await response.json();
                                if (result.success) {
                                    const val = result.text || 'No text';
                                    recognizedTexts.push(val);
                                    appendOcrTerminalLine('OCR OK', `T${tIdx + 1} → "${val}"`, 'ok');
                                    appendProgressiveResult(key, tIdx + 1, val, false);
                                } else {
                                    const errMsg = result.error || 'OCR recognition error';
                                    recognizedTexts.push('Error');
                                    appendOcrTerminalLine('ERROR', `T${tIdx + 1} → ${errMsg}`, 'err');
                                    appendProgressiveResult(key, tIdx + 1, errMsg, true);
                                }
                            } catch (err) {
                                const errMsg = err.message || 'Network error';
                                recognizedTexts.push('Error');
                                appendOcrTerminalLine('ERROR', `T${tIdx + 1} → ${errMsg}`, 'err');
                                appendProgressiveResult(key, tIdx + 1, errMsg, true);
                            }

                            processedTexts++;
                            const percent = Math.round((processedTexts / totalTexts) * 100);
                            if (progressBar) {
                                progressBar.style.width = percent + '%';
                                progressBar.textContent = percent >= 8 ? percent + '%' : '';
                            }
                            if (progressPercent) progressPercent.textContent = percent + '%';
                            if (progressText) {
                                progressText.innerHTML = `<i class="bi bi-arrow-repeat spin mr-1"></i> Processing box ${processedTexts} of ${totalTexts}...`;
                            }
                            if (countEl) countEl.textContent = `${processedTexts} / ${totalTexts} text boxes`;
                        }

                        state.ocrResults[key] = recognizedTexts;
                    }
                }

                appendOcrTerminalLine('COMPLETE', `All ${totalTexts} text areas transcribed successfully!`, 'ok');

                if (progressText) {
                    progressText.innerHTML = '<i class="bi bi-check-circle-fill mr-1" style="color: var(--teal);"></i> Extraction complete!';
                }
                if (progressPercent) progressPercent.textContent = '100%';
                if (progressBar) {
                    progressBar.style.width = '100%';
                    progressBar.textContent = '100%';
                }

                if (currentProject) {
                    await saveOCRResultsToProject();
                }

                if (proceedCleanBtn) proceedCleanBtn.classList.remove('hidden');
                enableTab('clean');
                success = true;

            } catch (err) {
                appendOcrTerminalLine('ERROR', `Fatal OCR failure: ${err.message}`, 'err');
            } finally {
                startBtn.dataset.running = 'false';
                startBtn.disabled = false;

                if (success) {
                    startBtn.innerHTML = '<i class="bi bi-arrow-repeat text-base"></i> Re-run OCR Processing';
                    if (skipBtn) skipBtn.classList.add('hidden');
                    if (visual) {
                        visual.innerHTML = '<div class="w-14 h-14 rounded-full flex items-center justify-center" style="background: rgba(13, 148, 136, 0.1); color: var(--teal);"><i class="bi bi-check-lg text-3xl"></i></div>';
                    }
                    if (headline) headline.textContent = 'Text Extraction Complete!';
                    if (subtitle) subtitle.textContent = `All ${totalTexts} text boxes transcribed and saved to the project catalog.`;
                    if (liveStepPill) liveStepPill.classList.add('hidden');
                    if (resultsSummary) resultsSummary.classList.remove('hidden');
                    if (actionsRow) actionsRow.classList.remove('hidden');
                    if (proceedCleanBtn) proceedCleanBtn.classList.remove('hidden');
                    const copyBtn = document.getElementById('copyOcrResultsBtn');
                    if (copyBtn) {
                        const labelSpan = copyBtn.querySelector('span');
                        if (labelSpan) labelSpan.textContent = 'Copy Extracted Texts';
                    }
                } else {
                    startBtn.innerHTML = '<i class="bi bi-play-fill text-base"></i> Retry OCR Processing';
                    if (skipBtn) {
                        skipBtn.classList.remove('hidden');
                        skipBtn.disabled = false;
                    }
                    if (proceedCleanBtn) proceedCleanBtn.classList.add('hidden');
                    if (visual) {
                        visual.innerHTML = '<div class="w-14 h-14 rounded-full flex items-center justify-center" style="background: rgba(239, 68, 68, 0.1); color: #dc2626;"><i class="bi bi-exclamation-triangle text-2xl"></i></div>';
                    }
                    if (headline) headline.textContent = 'Text Extraction Interrupted';
                    if (subtitle) subtitle.textContent = 'An error occurred during OCR recognition. Click Retry or check details.';
                    if (liveStepPill) liveStepPill.classList.add('hidden');
                    if (actionsRow) actionsRow.classList.remove('hidden');
                    const copyBtn = document.getElementById('copyOcrResultsBtn');
                    if (copyBtn) {
                        const labelSpan = copyBtn.querySelector('span');
                        if (labelSpan) labelSpan.textContent = 'Copy Diagnostic Details';
                    }
                }
            }
        });

        // Save OCR results to project
        async function saveOCRResultsToProject() {
            if (!currentProject) return;

            const results = [];

            for (const key in state.ocrResults) {
                results.push({
                    key: key,
                    texts: state.ocrResults[key]
                });
            }

            try {
                const response = await fetch(`/api/project/${currentProject.project_id}/save_ocr_results`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ results: results })
                });

                const data = await response.json();

                if (data.success) {
                    console.log('OCR results saved to project');
                } else {
                    console.error('Failed to save OCR results:', data.error);
                }
            } catch (error) {
                console.error('Error saving OCR results:', error);
            }
        }

        // Skip OCR button - proceed directly to manual input
        document.getElementById('skipOcrBtn').addEventListener('click', async () => {
            const confirmed = await showConfirmDialog({
                title: 'Skip OCR',
                message: 'Skip automated OCR processing and input vessel text manually in the Review tab?',
                confirmText: 'Skip OCR',
                cancelText: 'Cancel',
                type: 'info',
                icon: 'bi-pencil-square'
            });

            if (confirmed) {
                appendOcrTerminalLine('SKIPPED', 'Automated OCR skipped by user. Manual transcription enabled in Review tab.', 'skip');

                // Hide skip button once clicked, show proceed clean
                const skipBtn = document.getElementById('skipOcrBtn');
                if (skipBtn) skipBtn.classList.add('hidden');
                const proceedBtn = document.getElementById('proceedCleanBtn');
                if (proceedBtn) proceedBtn.classList.remove('hidden');
                enableTab('clean');

                const headline = document.getElementById('ocrCenterHeadline');
                const subtitle = document.getElementById('ocrCenterSubtitle');
                const visual = document.getElementById('ocrCenterVisual');
                if (visual) {
                    visual.innerHTML = '<div class="w-14 h-14 rounded-full flex items-center justify-center" style="background: var(--bg-surface); color: var(--text-muted);"><i class="bi bi-pencil-square text-2xl"></i></div>';
                }
                if (headline) headline.textContent = 'OCR Processing Skipped';
                if (subtitle) subtitle.textContent = 'Automated text recognition was skipped. You can input vessel labels manually in the Review stage, or click "Start OCR Processing" at any time.';
            }
        });

        document.getElementById('proceedCleanBtn').addEventListener('click', () => {
            switchTab('clean');
        });

        // Load data for Clean tab when switching to it
        async function loadCleanTabData(projectId) {
            if (!projectId) return;

            console.log('[Clean] Loading Clean tab data...');

            // Show loading overlay
            const overlay = document.getElementById('cleanTabLoadingOverlay');
            const loadingText = document.getElementById('cleanTabLoadingText');

            overlay.classList.remove('hidden');
            loadingText.textContent = 'Preparing to load drawings...';

            try {
                // Count total DRAWINGS to load (not images)
                let totalDrawings = 0;
                for (const imgName in state.annotations) {
                    totalDrawings += state.annotations[imgName].drawings.length;
                }

                console.log(`[Clean] Total drawings to load: ${totalDrawings}`);

                // Get ALL images that need to be loaded
                const allImageNames = Object.keys(state.annotations);
                const totalImages = allImageNames.length;
                let loadedImages = 0;

                // Update initial progress text with drawing count
                loadingText.textContent = `Loading images for ${totalDrawings} drawings (0 / ${totalImages})...`;

                // 1. Load full-res images if not already loaded
                for (const imgName of allImageNames) {
                    if (!state.fullResImages[imgName]) {
                        console.log(`Loading full-res image: ${imgName}`);
                        const imgElement = new Image();
                        imgElement.crossOrigin = 'anonymous';

                        await new Promise((resolve, reject) => {
                            imgElement.onload = () => {
                                state.fullResImages[imgName] = imgElement;
                                console.log(`[Clean] Loaded: ${imgName}`);

                                // Update progress
                                loadedImages++;
                                loadingText.textContent = `Loading images for ${totalDrawings} drawings (${loadedImages} / ${totalImages})...`;

                                resolve();
                            };
                            imgElement.onerror = () => {
                                console.error(`[Clean] Failed to load: ${imgName}`);
                                reject();
                            };
                            imgElement.src = `/api/project/${projectId}/image/${encodeURIComponent(imgName)}`;
                        });
                    } else {
                        // Already loaded, just increment counter
                        loadedImages++;
                        loadingText.textContent = `Loading images for ${totalDrawings} drawings (${loadedImages} / ${totalImages})...`;
                        console.log(`[Clean] Already loaded: ${imgName}`);
                    }
                }

                // 2. Load cleaned drawings from folder
                loadingText.textContent = 'Loading cleaned drawings...';

                await loadCleanedDrawingsFromProject(projectId);

                loadingText.textContent = 'Initializing canvas...';

                // 3. Initialize Clean tab with first item
                if (Object.keys(state.annotations).length > 0) {
                    console.log('[Clean] Clean tab data loaded, initializing canvas...');

                    // Hide overlay BEFORE calling loadClean to avoid UI freeze
                    setTimeout(() => {
                        overlay.classList.add('hidden');
                    }, 100);

                    try {
                        loadClean(0);
                        console.log('[Clean] Canvas initialized successfully');
                    } catch (err) {
                        console.error('[Clean] Error initializing clean canvas:', err);
                        alert('Error initializing canvas: ' + err.message);
                    }
                } else {
                    console.warn('[Clean] No annotations found for Clean tab');
                    setTimeout(() => {
                        overlay.classList.add('hidden');
                    }, 300);
                }

            } catch (error) {
                console.error('[Clean] Error loading Clean tab data:', error);
                loadingText.textContent = 'Error loading data: ' + error.message;
                // Hide overlay even on error
                setTimeout(() => {
                    overlay.classList.add('hidden');
                }, 2000); // Show error for 2 seconds
            }
        }

        // TAB 4: Clean Drawings with Eraser & Straighten Tool (Upgraded with Preview, Continuous Strokes, Zoom & Presets)
        let cleanIndex = 0;
        let cleanItems = [];
        let cleanCanvas, cleanCtx;
        let cleanSourceCanvas = null, cleanSourceCtx = null;
        let currentStraightenAngle = 0;
        let isGridActive = false;
        let isErasing = false;
        let eraserMode = false;
        let eraserSize = 20;
        let undoStack = [];
        let cleanDisplayScale = 1; // Base display scale
        let cleanZoom = 1.0; // Current zoom level (0.5 to 3.0)
        let lastCleanCoords = null;
        let cleanAutoSaveTimer = null;

        function loadClean(idx) {
            cleanItems = [];

            for (const imgName in state.annotations) {
                const ann = state.annotations[imgName];
                ann.drawings.forEach((d, dIdx) => {
                    const key = `${imgName}_d${dIdx + 1}`;
                    cleanItems.push({
                        imgName,
                        drawingIdx: dIdx,
                        drawing: d,
                        metadata: ann.metadata,
                        key
                    });
                });
            }

            if (cleanItems.length === 0) {
                console.warn('[Clean] No drawings to clean');
                return;
            }

            cleanIndex = idx;
            const item = cleanItems[cleanIndex];
            const imgElement = state.fullResImages[item.imgName];

            // Check if image is loaded
            if (!imgElement) {
                console.error(`[Clean] Image not loaded for ${item.imgName}`);
                return;
            }

            // Setup canvas for erasing
            if (!cleanCanvas) {
                cleanCanvas = document.getElementById('cleanCanvas');
                cleanCtx = cleanCanvas.getContext('2d');

                cleanCanvas.addEventListener('mousedown', handleCleanMouseDown);
                cleanCanvas.addEventListener('mousemove', handleCleanMouseMove);
                cleanCanvas.addEventListener('mouseup', handleCleanMouseUp);
                cleanCanvas.addEventListener('mouseleave', () => {
                    const preview = document.getElementById('cleanBrushPreview');
                    if (preview) preview.style.display = 'none';
                    handleCleanMouseUp();
                });
                window.addEventListener('mouseup', () => {
                    if (isErasing) handleCleanMouseUp();
                });
                cleanCanvas.addEventListener('mouseenter', (e) => {
                    if (eraserMode) {
                        updateBrushPreview(e);
                    }
                });

                // Mouse wheel: brush size when eraser is active, zoom when Ctrl/Alt is held
                cleanCanvas.addEventListener('wheel', (e) => {
                    if (e.altKey || e.ctrlKey) {
                        e.preventDefault();
                        if (e.deltaY < 0) {
                            setCleanZoom(cleanZoom + 0.2);
                        } else {
                            setCleanZoom(cleanZoom - 0.2);
                        }
                    } else if (eraserMode) {
                        e.preventDefault();
                        const step = e.deltaY < 0 ? 3 : -3;
                        updateEraserSize(eraserSize + step);
                    }
                }, { passive: false });
            }

            currentStraightenAngle = 0;
            resetStraightenSliderUI();

            // Check if a cleaned version exists in state
            if (state.cleanedDrawings[item.key]?.imageData) {
                // Load the cleaned version
                const cleanedImg = new Image();
                cleanedImg.onload = () => {
                    cleanCanvas.width = cleanedImg.naturalWidth || cleanedImg.width;
                    cleanCanvas.height = cleanedImg.naturalHeight || cleanedImg.height;
                    cleanCtx.fillStyle = '#ffffff';
                    cleanCtx.fillRect(0, 0, cleanCanvas.width, cleanCanvas.height);
                    cleanCtx.drawImage(cleanedImg, 0, 0);
                    initCleanSourceCanvas();
                    resetStraightenSliderUI();
                    applyCleanCanvasDimensions();
                    if (isGridActive) drawCleanGrid();
                    // Save to undo stack
                    undoStack = [cleanCanvas.toDataURL()];

                    // AUTO-RECOVERY GUARDRAIL:
                    // If canvas was previously blown up into an enormous whitespace rectangle
                    // (e.g. natural dimensions > 1.25x original detection crop), auto-trim immediately!
                    const origMax = Math.max(item.drawing.w, item.drawing.h);
                    const curMax = Math.max(cleanCanvas.width, cleanCanvas.height);
                    if (curMax > origMax * 1.25) {
                        console.log(`[Clean] Auto-recovering bloated canvas (${cleanCanvas.width}x${cleanCanvas.height} vs original max ${origMax})`);
                        trimCanvasWhitespace(24);
                    }
                };
                cleanedImg.src = state.cleanedDrawings[item.key].imageData;
                console.log('Loading cleaned version for:', item.key);
            } else {
                // Draw original image at full resolution
                cleanCanvas.width = item.drawing.w;
                cleanCanvas.height = item.drawing.h;
                cleanCtx.fillStyle = '#ffffff';
                cleanCtx.fillRect(0, 0, cleanCanvas.width, cleanCanvas.height);
                cleanCtx.drawImage(imgElement, item.drawing.x, item.drawing.y, item.drawing.w, item.drawing.h, 0, 0, item.drawing.w, item.drawing.h);
                initCleanSourceCanvas();
                resetStraightenSliderUI();
                applyCleanCanvasDimensions();
                if (isGridActive) drawCleanGrid();
                // Save initial state for undo
                undoStack = [cleanCanvas.toDataURL()];
                console.log('Loading original version for:', item.key);
            }

            // Apply sizing and zoom
            cleanZoom = 1.0;
            applyCleanCanvasDimensions();

            // Set metadata labels
            document.getElementById('cleanPlate').textContent = item.metadata.tableName || item.imgName;
            document.getElementById('cleanDrawingId').textContent = `Drawing D${item.drawingIdx + 1}`;
            const dimEl = document.getElementById('cleanDimensions');
            if (dimEl) {
                const curW = cleanCanvas ? cleanCanvas.width : item.drawing.w;
                const curH = cleanCanvas ? cleanCanvas.height : item.drawing.h;
                dimEl.textContent = `${Math.round(curW)} × ${Math.round(curH)} px`;
            }
            document.getElementById('cleanIndicator').textContent = `${cleanIndex + 1} / ${cleanItems.length}`;

            updateEraserSize(eraserSize);
            eraserMode = false;
            updateCleanUI();
            renderCleanThumbnails();
        }

        function applyCleanCanvasDimensions() {
            const item = cleanItems[cleanIndex];
            if (!item || !cleanCanvas) return;

            const outerWrapper = document.getElementById('cleanCanvasOuterWrapper');
            const availableW = outerWrapper && outerWrapper.clientWidth > 100 ? outerWrapper.clientWidth - 48 : 800;
            const maxDisplayWidth = Math.max(700, Math.min(1050, availableW));

            const curW = cleanCanvas.width || item.drawing.w;
            const curH = cleanCanvas.height || item.drawing.h;
            const baseScale = Math.min(1, maxDisplayWidth / curW);
            cleanDisplayScale = baseScale * cleanZoom;

            const dispW = Math.round(curW * cleanDisplayScale);
            const dispH = Math.round(curH * cleanDisplayScale);

            cleanCanvas.style.width = dispW + 'px';
            cleanCanvas.style.height = dispH + 'px';

            const zoomDisplay = document.getElementById('cleanZoomDisplay');
            if (zoomDisplay) {
                zoomDisplay.textContent = Math.round(cleanZoom * 100) + '%';
            }

            const dimEl = document.getElementById('cleanDimensions');
            if (dimEl) {
                dimEl.textContent = `${Math.round(curW)} × ${Math.round(curH)} px`;
            }

            resizeCleanGridOverlay();
        }

        // STRAIGHTEN & ROTATION UTILITIES
        function initCleanSourceCanvas() {
            if (!cleanSourceCanvas) {
                cleanSourceCanvas = document.createElement('canvas');
                cleanSourceCtx = cleanSourceCanvas.getContext('2d');
            }
            if (!cleanCanvas) return;
            cleanSourceCanvas.width = cleanCanvas.width;
            cleanSourceCanvas.height = cleanCanvas.height;
            cleanSourceCtx.fillStyle = '#ffffff';
            cleanSourceCtx.fillRect(0, 0, cleanCanvas.width, cleanCanvas.height);
            cleanSourceCtx.drawImage(cleanCanvas, 0, 0);
        }

        function applyStraightenAngle(angle) {
            if (!cleanSourceCanvas || !cleanCanvas) return;
            currentStraightenAngle = Math.round(angle * 10) / 10;

            // Update UI readout
            const display = document.getElementById('straightenAngleDisplay');
            if (display) {
                const sign = currentStraightenAngle > 0 ? '+' : '';
                display.textContent = `${sign}${currentStraightenAngle.toFixed(1)}°`;
            }

            const slider = document.getElementById('straightenSlider');
            if (slider && Math.abs(parseFloat(slider.value) - currentStraightenAngle) > 0.05) {
                slider.value = currentStraightenAngle;
            }
            updateStraightenSliderFill(currentStraightenAngle);

            const srcW = cleanSourceCanvas.width;
            const srcH = cleanSourceCanvas.height;

            if (Math.abs(currentStraightenAngle) < 0.001) {
                cleanCanvas.width = srcW;
                cleanCanvas.height = srcH;
                cleanCtx.fillStyle = '#ffffff';
                cleanCtx.fillRect(0, 0, srcW, srcH);
                cleanCtx.drawImage(cleanSourceCanvas, 0, 0);
            } else {
                const rad = (currentStraightenAngle * Math.PI) / 180;
                const sin = Math.abs(Math.sin(rad));
                const cos = Math.abs(Math.cos(rad));

                // 1. Dynamic Bounding Box: Expand canvas dynamically so wide or tilted vessels never get clipped
                let newW = Math.ceil(srcW * cos + srcH * sin);
                let newH = Math.ceil(srcH * cos + srcW * sin);

                // 2. Geometric Guardrail: Cap maximum dimension by the physical diagonal of the vessel
                const item = cleanItems[cleanIndex];
                if (item && item.drawing) {
                    const origDiag = Math.ceil(Math.sqrt(item.drawing.w * item.drawing.w + item.drawing.h * item.drawing.h));
                    const maxAllowed = Math.round(origDiag * 1.15) + 32;
                    newW = Math.min(newW, maxAllowed);
                    newH = Math.min(newH, maxAllowed);
                }

                cleanCanvas.width = newW;
                cleanCanvas.height = newH;
                cleanCtx.fillStyle = '#ffffff';
                cleanCtx.fillRect(0, 0, newW, newH);

                cleanCtx.imageSmoothingEnabled = true;
                cleanCtx.imageSmoothingQuality = 'high';

                cleanCtx.save();
                cleanCtx.translate(newW / 2, newH / 2);
                cleanCtx.rotate(rad);
                cleanCtx.drawImage(cleanSourceCanvas, -srcW / 2, -srcH / 2);
                cleanCtx.restore();
            }

            applyCleanCanvasDimensions();
            if (isGridActive) drawCleanGrid();
        }

        function updateStraightenSliderFill(val) {
            const slider = document.getElementById('straightenSlider');
            if (!slider) return;
            const min = -45, max = 45;
            const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
            if (val > 0.05) {
                slider.style.background = `linear-gradient(to right, #e8e3d8 0%, #e8e3d8 50%, #c2410c 50%, #c2410c ${pct}%, #e8e3d8 ${pct}%, #e8e3d8 100%)`;
            } else if (val < -0.05) {
                slider.style.background = `linear-gradient(to right, #e8e3d8 0%, #e8e3d8 ${pct}%, #c2410c ${pct}%, #c2410c 50%, #e8e3d8 50%, #e8e3d8 100%)`;
            } else {
                slider.style.background = '#e8e3d8';
            }
        }

        function resetStraightenSliderUI() {
            currentStraightenAngle = 0;
            const slider = document.getElementById('straightenSlider');
            if (slider) {
                slider.value = 0;
                slider.style.background = '#e8e3d8';
            }
            const display = document.getElementById('straightenAngleDisplay');
            if (display) display.textContent = '0.0°';
        }

        function commitCleanRotation() {
            if (Math.abs(currentStraightenAngle) > 0.001 && cleanCanvas) {
                // Auto-trim empty corner wedges around the vessel so whitespace never compounds
                const trimmed = trimCanvasWhitespace(24);
                if (!trimmed) {
                    initCleanSourceCanvas();
                    resetStraightenSliderUI();
                    undoStack.push(cleanCanvas.toDataURL());
                    if (undoStack.length > 30) undoStack.shift();
                    triggerCleanAutoSave();
                    renderCleanThumbnails();
                }
            }
        }

        // Auto-Fit / Trim Whitespace guardrail: crops away runaway white borders around the vessel
        function trimCanvasWhitespace(padding = 24) {
            if (!cleanCanvas || !cleanCtx) return false;
            const w = cleanCanvas.width;
            const h = cleanCanvas.height;
            if (w <= 20 || h <= 20) return false;

            currentStraightenAngle = 0;

            let imgData;
            try {
                imgData = cleanCtx.getImageData(0, 0, w, h);
            } catch (e) {
                console.error('[Clean] Cannot get image data for trimming:', e);
                return false;
            }
            const data = imgData.data;
            const threshold = 245; // Treat pixels with R,G,B >= 245 as blank background
            let minX = w, maxX = -1, minY = h, maxY = -1;

            // 1. Scan top bound (minY)
            topLoop: for (let y = 0; y < h; y++) {
                const row = y * w * 4;
                for (let x = 0; x < w; x++) {
                    const idx = row + x * 4;
                    if (data[idx + 3] > 50 && (data[idx] < threshold || data[idx + 1] < threshold || data[idx + 2] < threshold)) {
                        minY = y;
                        break topLoop;
                    }
                }
            }

            if (minY === h) {
                showCleanAutosaveBadge('No drawing found');
                return false;
            }

            // 2. Scan bottom bound (maxY)
            bottomLoop: for (let y = h - 1; y >= minY; y--) {
                const row = y * w * 4;
                for (let x = 0; x < w; x++) {
                    const idx = row + x * 4;
                    if (data[idx + 3] > 50 && (data[idx] < threshold || data[idx + 1] < threshold || data[idx + 2] < threshold)) {
                        maxY = y;
                        break bottomLoop;
                    }
                }
            }

            // 3. Scan left bound (minX) within [minY, maxY]
            leftLoop: for (let x = 0; x < w; x++) {
                for (let y = minY; y <= maxY; y++) {
                    const idx = (y * w + x) * 4;
                    if (data[idx + 3] > 50 && (data[idx] < threshold || data[idx + 1] < threshold || data[idx + 2] < threshold)) {
                        minX = x;
                        break leftLoop;
                    }
                }
            }

            // 4. Scan right bound (maxX) within [minY, maxY]
            rightLoop: for (let x = w - 1; x >= minX; x--) {
                for (let y = minY; y <= maxY; y++) {
                    const idx = (y * w + x) * 4;
                    if (data[idx + 3] > 50 && (data[idx] < threshold || data[idx + 1] < threshold || data[idx + 2] < threshold)) {
                        maxX = x;
                        break rightLoop;
                    }
                }
            }

            const cropX = Math.max(0, minX - padding);
            const cropY = Math.max(0, minY - padding);
            const cropR = Math.min(w, maxX + 1 + padding);
            const cropB = Math.min(h, maxY + 1 + padding);
            const cropW = cropR - cropX;
            const cropH = cropB - cropY;

            // Skip if canvas is already compact (less than 12px change)
            if (cropX <= 6 && cropY <= 6 && (w - cropR) <= 6 && (h - cropB) <= 6) {
                showCleanAutosaveBadge('Already fitted');
                return false;
            }

            const temp = document.createElement('canvas');
            temp.width = cropW;
            temp.height = cropH;
            const tempCtx = temp.getContext('2d');
            tempCtx.fillStyle = '#ffffff';
            tempCtx.fillRect(0, 0, cropW, cropH);
            tempCtx.drawImage(cleanCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

            cleanCanvas.width = cropW;
            cleanCanvas.height = cropH;
            cleanCtx.fillStyle = '#ffffff';
            cleanCtx.fillRect(0, 0, cropW, cropH);
            cleanCtx.drawImage(temp, 0, 0);

            initCleanSourceCanvas();
            resetStraightenSliderUI();
            applyCleanCanvasDimensions();
            if (isGridActive) drawCleanGrid();

            undoStack.push(cleanCanvas.toDataURL());
            if (undoStack.length > 30) undoStack.shift();
            triggerCleanAutoSave();
            renderCleanThumbnails();
            showCleanAutosaveBadge('Auto-fitted');
            return true;
        }

        function stepStraightenAngle(delta) {
            let newAngle = Math.round((currentStraightenAngle + delta) * 10) / 10;
            newAngle = Math.max(-45, Math.min(45, newAngle));
            applyStraightenAngle(newAngle);
            undoStack.push(cleanCanvas.toDataURL());
            if (undoStack.length > 30) undoStack.shift();
            triggerCleanAutoSave();
            renderCleanThumbnails();
        }

        function rotateCleanCanvas90(direction) {
            if (!cleanCanvas) return;
            commitCleanRotation();

            const srcW = cleanCanvas.width;
            const srcH = cleanCanvas.height;
            const newW = srcH;
            const newH = srcW;

            const temp = document.createElement('canvas');
            temp.width = newW;
            temp.height = newH;
            const tempCtx = temp.getContext('2d');

            tempCtx.fillStyle = '#ffffff';
            tempCtx.fillRect(0, 0, newW, newH);
            tempCtx.imageSmoothingEnabled = true;
            tempCtx.imageSmoothingQuality = 'high';

            tempCtx.save();
            tempCtx.translate(newW / 2, newH / 2);
            tempCtx.rotate((direction * 90 * Math.PI) / 180);
            tempCtx.drawImage(cleanCanvas, -srcW / 2, -srcH / 2);
            tempCtx.restore();

            cleanCanvas.width = newW;
            cleanCanvas.height = newH;
            cleanCtx.drawImage(temp, 0, 0);

            initCleanSourceCanvas();
            resetStraightenSliderUI();
            applyCleanCanvasDimensions();
            if (isGridActive) drawCleanGrid();

            undoStack.push(cleanCanvas.toDataURL());
            if (undoStack.length > 30) undoStack.shift();
            triggerCleanAutoSave();
            renderCleanThumbnails();
        }

        function toggleCleanGrid() {
            isGridActive = !isGridActive;
            updateCleanGridUI();
        }

        function updateCleanGridUI() {
            const gridOverlay = document.getElementById('cleanGridOverlay');
            const toggleBtn = document.getElementById('toggleGridBtn');
            const outerWrapper = document.getElementById('cleanCanvasOuterWrapper');
            if (!gridOverlay || !toggleBtn) return;

            if (isGridActive) {
                gridOverlay.style.display = 'block';
                toggleBtn.classList.add('btn-grid-active');
                toggleBtn.classList.remove('btn-secondary');
                if (outerWrapper) outerWrapper.classList.add('hide-bg-grid');
                drawCleanGrid();
            } else {
                gridOverlay.style.display = 'none';
                toggleBtn.classList.remove('btn-grid-active');
                toggleBtn.classList.add('btn-secondary');
                if (outerWrapper) outerWrapper.classList.remove('hide-bg-grid');
            }
        }

        function resizeCleanGridOverlay() {
            if (isGridActive) drawCleanGrid();
        }

        function drawCleanGrid() {
            const gridOverlay = document.getElementById('cleanGridOverlay');
            const outerWrapper = document.getElementById('cleanCanvasOuterWrapper');
            if (!gridOverlay || !outerWrapper || !isGridActive) return;

            // Use client dimensions of outerWrapper so grid matches visible viewport
            const w = outerWrapper.clientWidth;
            const h = outerWrapper.clientHeight;
            if (w === 0 || h === 0) return;

            if (gridOverlay.width !== w || gridOverlay.height !== h) {
                gridOverlay.width = w;
                gridOverlay.height = h;
            }

            const gctx = gridOverlay.getContext('2d');
            gctx.clearRect(0, 0, w, h);

            // Step size is constant: EXACTLY 30px (never shifts or resizes with slider)
            const step = 30;

            // Find center of cleanCanvas on screen relative to outerWrapper
            let midX = Math.round(w / 2);
            let midY = Math.round(h / 2);

            if (cleanCanvas) {
                const cRect = cleanCanvas.getBoundingClientRect();
                const wRect = outerWrapper.getBoundingClientRect();
                midX = Math.round(cRect.left - wRect.left + cRect.width / 2);
                midY = Math.round(cRect.top - wRect.top + cRect.height / 2);
            }

            // 1. Stable Blue Grid Lines (constant spacing, never mutating)
            gctx.lineWidth = 1;
            gctx.strokeStyle = 'rgba(2, 132, 199, 0.35)';

            // Horizontal lines extending upward and downward from midY
            for (let y = midY; y >= 0; y -= step) {
                gctx.beginPath();
                gctx.moveTo(0, Math.round(y) + 0.5);
                gctx.lineTo(w, Math.round(y) + 0.5);
                gctx.stroke();
            }
            for (let y = midY + step; y < h; y += step) {
                gctx.beginPath();
                gctx.moveTo(0, Math.round(y) + 0.5);
                gctx.lineTo(w, Math.round(y) + 0.5);
                gctx.stroke();
            }

            // Vertical lines extending left and right from midX
            for (let x = midX; x >= 0; x -= step) {
                gctx.beginPath();
                gctx.moveTo(Math.round(x) + 0.5, 0);
                gctx.lineTo(Math.round(x) + 0.5, h);
                gctx.stroke();
            }
            for (let x = midX + step; x < w; x += step) {
                gctx.beginPath();
                gctx.moveTo(Math.round(x) + 0.5, 0);
                gctx.lineTo(Math.round(x) + 0.5, h);
                gctx.stroke();
            }

            // 2. Central Reference Lines (Rim level baseline & Symmetry axis)
            gctx.lineWidth = 1.5;
            gctx.strokeStyle = 'rgba(194, 65, 12, 0.85)';
            gctx.setLineDash([6, 4]);

            // Central horizontal line
            gctx.beginPath();
            gctx.moveTo(0, Math.round(midY) + 0.5);
            gctx.lineTo(w, Math.round(midY) + 0.5);
            gctx.stroke();

            // Central vertical line
            gctx.beginPath();
            gctx.moveTo(Math.round(midX) + 0.5, 0);
            gctx.lineTo(Math.round(midX) + 0.5, h);
            gctx.stroke();

            gctx.setLineDash([]);
        }

        function setCleanZoom(newZoom) {
            cleanZoom = Math.max(0.5, Math.min(3.0, Math.round(newZoom * 100) / 100));
            applyCleanCanvasDimensions();
            // Also refresh brush preview if visible
            const preview = document.getElementById('cleanBrushPreview');
            if (preview && preview.style.display !== 'none' && cleanCanvas) {
                const rect = cleanCanvas.getBoundingClientRect();
                const scale = rect.width / cleanCanvas.width;
                const screenDiameter = Math.max(4, eraserSize * scale);
                preview.style.width = screenDiameter + 'px';
                preview.style.height = screenDiameter + 'px';
            }
        }

        function updateBrushPreview(e) {
            const preview = document.getElementById('cleanBrushPreview');
            if (!preview || !cleanCanvas) return;

            if (!eraserMode) {
                preview.style.display = 'none';
                cleanCanvas.style.cursor = 'default';
                return;
            }

            const rect = cleanCanvas.getBoundingClientRect();
            // Check if cursor is within canvas bounds
            const isOver = (
                e.clientX >= rect.left &&
                e.clientX <= rect.right &&
                e.clientY >= rect.top &&
                e.clientY <= rect.bottom
            );

            if (!isOver) {
                preview.style.display = 'none';
                cleanCanvas.style.cursor = 'default';
                return;
            }

            // Cursor is over canvas in eraser mode: hide native arrow and display custom brush ring
            cleanCanvas.style.cursor = 'none';

            // Calculate exact screen diameter matching canvas rendering scale
            const scale = rect.width / cleanCanvas.width;
            const screenDiameter = Math.max(6, Math.round(eraserSize * scale));

            preview.style.width = screenDiameter + 'px';
            preview.style.height = screenDiameter + 'px';
            preview.style.left = e.clientX + 'px';
            preview.style.top = e.clientY + 'px';
            preview.style.display = 'block';
        }

        function updateEraserSize(newSize) {
            eraserSize = Math.max(2, Math.min(150, Math.round(newSize)));
            const sizeSlider = document.getElementById('eraserSize');
            if (sizeSlider) {
                sizeSlider.value = eraserSize;
                const min = parseInt(sizeSlider.min) || 4;
                const max = parseInt(sizeSlider.max) || 120;
                const pct = Math.max(0, Math.min(100, ((eraserSize - min) / (max - min)) * 100));
                sizeSlider.style.background = `linear-gradient(to right, #c2410c 0%, #c2410c ${pct}%, #e8e3d8 ${pct}%, #e8e3d8 100%)`;
            }

            const sizeDisplay = document.getElementById('eraserSizeDisplay');
            if (sizeDisplay) sizeDisplay.textContent = eraserSize + 'px';

            // Update preset buttons and pills active highlight
            document.querySelectorAll('.preset-pill, .preset-eraser-btn').forEach(btn => {
                if (parseInt(btn.dataset.size) === eraserSize) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });

            // Update swatch lens circle
            const indicator = document.getElementById('eraserSizeIndicator');
            if (indicator) {
                const dotPx = Math.max(4, Math.min(26, Math.round((eraserSize / 120) * 22 + 4)));
                indicator.style.width = dotPx + 'px';
                indicator.style.height = dotPx + 'px';
            }

            // Update brush preview if visible
            const preview = document.getElementById('cleanBrushPreview');
            if (preview && preview.style.display !== 'none' && cleanCanvas) {
                const rect = cleanCanvas.getBoundingClientRect();
                const scale = rect.width / cleanCanvas.width;
                const screenDiameter = Math.max(6, Math.round(eraserSize * scale));
                preview.style.width = screenDiameter + 'px';
                preview.style.height = screenDiameter + 'px';
            }
        }

        // Render thumbnails grid for clean tab
        function renderCleanThumbnails() {
            const grid = document.getElementById('cleanThumbnailsGrid');
            if (!grid) return;

            // Count cleaned drawings
            const cleanedCount = cleanItems.filter(item => state.cleanedDrawings[item.key]?.cleaned).length;
            const counterEl = document.getElementById('cleanedCounter');
            if (counterEl) {
                counterEl.textContent = `${cleanedCount} / ${cleanItems.length} cleaned`;
            }

            grid.innerHTML = cleanItems.map((item, idx) => {
                const isCleaned = state.cleanedDrawings[item.key]?.cleaned || false;
                const isActive = idx === cleanIndex;

                // Generate thumbnail for this specific drawing crop
                let thumbDataUrl = '';
                const imgElement = state.fullResImages[item.imgName];
                if (imgElement && imgElement.complete) {
                    // Create temporary canvas for thumbnail
                    const tempCanvas = document.createElement('canvas');
                    const tempCtx = tempCanvas.getContext('2d');

                    // Calculate thumbnail dimensions (max 150px width, maintain aspect ratio)
                    const maxThumbWidth = 150;
                    const aspectRatio = item.drawing.h / item.drawing.w;
                    const thumbWidth = Math.min(maxThumbWidth, item.drawing.w);
                    const thumbHeight = thumbWidth * aspectRatio;

                    tempCanvas.width = thumbWidth;
                    tempCanvas.height = thumbHeight;

                    // Fill white background
                    tempCtx.fillStyle = 'white';
                    tempCtx.fillRect(0, 0, thumbWidth, thumbHeight);

                    // Draw cropped region (or cleaned version if available)
                    if (state.cleanedDrawings[item.key]?.imageData) {
                        const cached = new Image();
                        cached.src = state.cleanedDrawings[item.key].imageData;
                        if (cached.complete) {
                            tempCtx.drawImage(cached, 0, 0, thumbWidth, thumbHeight);
                        } else {
                            tempCtx.drawImage(
                                imgElement,
                                item.drawing.x, item.drawing.y, item.drawing.w, item.drawing.h,
                                0, 0, thumbWidth, thumbHeight
                            );
                        }
                    } else {
                        tempCtx.drawImage(
                            imgElement,
                            item.drawing.x, item.drawing.y, item.drawing.w, item.drawing.h,
                            0, 0, thumbWidth, thumbHeight
                        );
                    }

                    thumbDataUrl = tempCanvas.toDataURL('image/jpeg', 0.7);
                }

                return `
                    <div class="clean-thumbnail ${isActive ? 'is-active' : ''} ${isCleaned ? 'is-cleaned' : ''}" data-index="${idx}">
                        ${thumbDataUrl
                            ? `<div class="clean-thumb-image"><img src="${thumbDataUrl}" alt="D${item.drawingIdx + 1}"></div>`
                            : `<div class="clean-thumb-placeholder">Caricamento...</div>`
                        }
                        <span class="clean-thumb-badge">D${item.drawingIdx + 1}</span>
                        ${isCleaned ? `<span class="clean-thumb-check"><i class="bi bi-check-lg"></i></span>` : ''}
                        <p class="clean-thumb-label ${isCleaned ? 'cleaned' : ''}" title="${item.imgName}">${item.imgName.length > 12 ? item.imgName.substring(0, 12) + '…' : item.imgName}</p>
                    </div>
                `;
            }).join('');


            // Add click handlers
            grid.querySelectorAll('.clean-thumbnail').forEach(thumb => {
                thumb.addEventListener('click', async () => {
                    const targetIdx = parseInt(thumb.dataset.index);
                    if (targetIdx !== cleanIndex) {
                        commitCleanRotation();
                        // Auto-save current before switching
                        await saveCurrentCleanDrawing();
                        eraserMode = false;
                        loadClean(targetIdx);
                    }
                });
            });
        }

        // Auto-save current clean drawing
        async function saveCurrentCleanDrawing() {
            if (!cleanItems[cleanIndex] || !cleanCanvas) return;
            const item = cleanItems[cleanIndex];

            // Save the cleaned drawing back to state
            const cleanedImageData = cleanCanvas.toDataURL();
            if (!state.cleanedDrawings[item.key]) {
                state.cleanedDrawings[item.key] = {};
            }
            // Preserve the existing 'cleaned' flag — only Mark Clean button sets it to true
            const wasAlreadyCleaned = state.cleanedDrawings[item.key]?.cleaned || false;
            state.cleanedDrawings[item.key].cleaned = wasAlreadyCleaned;
            state.cleanedDrawings[item.key].imageData = cleanedImageData;


            // Save to project backend
            if (currentProject) {
                const baseImgName = item.imgName.replace(/\.[^/.]+$/, '');
                const cleanedFilename = `${baseImgName}_d${item.drawingIdx + 1}_cleaned.png`;

                try {
                    const response = await fetch(`/api/project/${currentProject.project_id}/save_cropped`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            image: cleanedImageData,
                            filename: cleanedFilename,
                            folder: 'cleaned_drawings'
                        })
                    });

                    const data = await response.json();

                    if (data.success) {
                        console.log('Cleaned drawing auto-saved:', cleanedFilename);
                    }
                } catch (error) {
                    console.error('Error auto-saving cleaned drawing:', error);
                }
            }
        }

        function triggerCleanAutoSave() {
            if (cleanAutoSaveTimer) clearTimeout(cleanAutoSaveTimer);
            cleanAutoSaveTimer = setTimeout(async () => {
                await saveCurrentCleanDrawing();
                showCleanAutosaveBadge('Saved');
            }, 600);
        }

        function showCleanAutosaveBadge(text = 'Saved') {
            const badge = document.getElementById('cleanAutosaveBadge');
            if (!badge) return;
            badge.innerHTML = `<i class="bi bi-check2-circle mr-1"></i> ${text}`;
            badge.classList.add('visible');
            setTimeout(() => {
                badge.classList.remove('visible');
            }, 2000);
        }

        function updateCleanUI() {
            const eraserBtn = document.getElementById('eraserBtn');
            const preview = document.getElementById('cleanBrushPreview');

            if (eraserMode) {
                eraserBtn.classList.add('eraser-btn-active');
                eraserBtn.classList.remove('btn-secondary');
                eraserBtn.innerHTML = '<i class="bi bi-eraser-fill"></i> <span>Eraser</span> <span class="text-xs font-mono opacity-70">(E)</span> <span class="pulse-dot"></span>';
                if (cleanCanvas) cleanCanvas.style.cursor = 'none';
            } else {
                eraserBtn.classList.remove('eraser-btn-active');
                eraserBtn.classList.add('btn-secondary');
                eraserBtn.innerHTML = '<i class="bi bi-eraser-fill"></i> <span>Eraser</span> <span class="text-xs font-mono opacity-70">(E)</span>';
                if (cleanCanvas) cleanCanvas.style.cursor = 'default';
                if (preview) preview.style.display = 'none';
            }
        }

        function getCleanCanvasCoords(e) {
            const rect = cleanCanvas.getBoundingClientRect();
            const scaleX = cleanCanvas.width / rect.width;
            const scaleY = cleanCanvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        function handleCleanMouseDown(e) {
            if (!eraserMode) return;
            isErasing = true;

            // Commit any unbaked rotation before erasing
            commitCleanRotation();

            const coords = getCleanCanvasCoords(e);
            lastCleanCoords = coords;
            erasePoint(coords.x, coords.y);
            updateBrushPreview(e);
        }

        function handleCleanMouseMove(e) {
            updateBrushPreview(e);
            if (!isErasing || !eraserMode) return;
            const coords = getCleanCanvasCoords(e);
            if (lastCleanCoords) {
                eraseStroke(lastCleanCoords.x, lastCleanCoords.y, coords.x, coords.y);
            } else {
                erasePoint(coords.x, coords.y);
            }
            lastCleanCoords = coords;
        }

        function handleCleanMouseUp() {
            if (isErasing) {
                isErasing = false;
                lastCleanCoords = null;
                initCleanSourceCanvas();
                undoStack.push(cleanCanvas.toDataURL());
                if (undoStack.length > 30) undoStack.shift();
                triggerCleanAutoSave();
                renderCleanThumbnails();
            }
        }

        function eraseStroke(x0, y0, x1, y1) {
            cleanCtx.save();
            cleanCtx.strokeStyle = '#ffffff';
            cleanCtx.fillStyle = '#ffffff';
            cleanCtx.lineWidth = eraserSize;
            cleanCtx.lineCap = 'round';
            cleanCtx.lineJoin = 'round';
            cleanCtx.beginPath();
            cleanCtx.moveTo(x0, y0);
            cleanCtx.lineTo(x1, y1);
            cleanCtx.stroke();
            cleanCtx.restore();
        }

        function erasePoint(x, y) {
            cleanCtx.save();
            cleanCtx.fillStyle = '#ffffff';
            cleanCtx.beginPath();
            cleanCtx.arc(x, y, eraserSize / 2, 0, Math.PI * 2);
            cleanCtx.fill();
            cleanCtx.restore();
        }

        function undoClean() {
            if (undoStack.length > 1) {
                undoStack.pop(); // Remove current state
                const previousState = undoStack[undoStack.length - 1];
                const img = new Image();
                img.onload = () => {
                    cleanCanvas.width = img.naturalWidth || img.width;
                    cleanCanvas.height = img.naturalHeight || img.height;
                    cleanCtx.clearRect(0, 0, cleanCanvas.width, cleanCanvas.height);
                    cleanCtx.fillStyle = '#ffffff';
                    cleanCtx.fillRect(0, 0, cleanCanvas.width, cleanCanvas.height);
                    cleanCtx.drawImage(img, 0, 0);
                    initCleanSourceCanvas();
                    resetStraightenSliderUI();
                    applyCleanCanvasDimensions();
                    if (isGridActive) drawCleanGrid();
                    triggerCleanAutoSave();
                    renderCleanThumbnails();
                };
                img.src = previousState;
            }
        }

        document.getElementById('eraserBtn').addEventListener('click', () => {
            eraserMode = !eraserMode;
            updateCleanUI();
        });

        // Preset buttons click handler (supporting both .preset-pill and .preset-eraser-btn)
        document.querySelectorAll('.preset-pill, .preset-eraser-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const sz = parseInt(btn.dataset.size);
                updateEraserSize(sz);
                if (!eraserMode) {
                    eraserMode = true;
                    updateCleanUI();
                }
            });
        });

        // Zoom buttons
        document.getElementById('cleanZoomInBtn').addEventListener('click', () => {
            setCleanZoom(cleanZoom + 0.25);
        });
        document.getElementById('cleanZoomOutBtn').addEventListener('click', () => {
            setCleanZoom(cleanZoom - 0.25);
        });
        document.getElementById('cleanZoomResetBtn').addEventListener('click', () => {
            setCleanZoom(1.0);
        });

        // Revert to original crop
        document.getElementById('resetCleanBtn').addEventListener('click', async () => {
            if (!cleanItems[cleanIndex]) return;
            const item = cleanItems[cleanIndex];
            const imgElement = state.fullResImages[item.imgName];
            if (!imgElement) return;

            const confirmed = await showConfirmDialog({
                title: 'Revert Drawing',
                message: 'Revert to the original unedited drawing? Any erasing and straightening done on this drawing will be discarded.',
                confirmText: 'Revert Drawing',
                cancelText: 'Keep Editing',
                type: 'danger',
                icon: 'bi-arrow-counterclockwise'
            });

            if (confirmed) {
                undoStack.push(cleanCanvas.toDataURL());
                cleanCanvas.width = item.drawing.w;
                cleanCanvas.height = item.drawing.h;
                cleanCtx.fillStyle = '#ffffff';
                cleanCtx.fillRect(0, 0, cleanCanvas.width, cleanCanvas.height);
                cleanCtx.drawImage(
                    imgElement,
                    item.drawing.x, item.drawing.y, item.drawing.w, item.drawing.h,
                    0, 0, item.drawing.w, item.drawing.h
                );
                initCleanSourceCanvas();
                resetStraightenSliderUI();
                applyCleanCanvasDimensions();
                if (isGridActive) drawCleanGrid();
                if (state.cleanedDrawings[item.key]) {
                    state.cleanedDrawings[item.key].cleaned = false;
                    delete state.cleanedDrawings[item.key].imageData;
                }
                renderCleanThumbnails();
                showCleanAutosaveBadge('Reverted');
            }
        });

        // Keyboard shortcuts for clean tab
        document.addEventListener('keydown', (e) => {
            // Only work in clean tab
            const activeTab = document.querySelector('.tab-content:not(.hidden)');
            if (!activeTab || activeTab.dataset.tab !== 'clean') return;

            // Don't trigger if typing in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            // E key - Toggle eraser
            if (e.key === 'e' || e.key === 'E') {
                e.preventDefault();
                eraserMode = !eraserMode;
                updateCleanUI();
            }

            // G key - Toggle alignment grid guide
            if (e.key === 'g' || e.key === 'G') {
                e.preventDefault();
                toggleCleanGrid();
            }

            // R key - Reset straighten rotation to 0°
            if (e.key === 'r' || e.key === 'R') {
                e.preventDefault();
                const resetBtn = document.getElementById('straightenResetBtn');
                if (resetBtn) resetBtn.click();
            }

            // M key - Mark as clean
            if (e.key === 'm' || e.key === 'M') {
                e.preventDefault();
                document.getElementById('markCleanBtn').click();
            }

            // Ctrl+Z or Cmd+Z - Undo
            if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
                e.preventDefault();
                undoClean();
            }

            // [ and ] - Eraser size adjust
            if (e.key === '[') {
                e.preventDefault();
                updateEraserSize(eraserSize - 5);
            }
            if (e.key === ']') {
                e.preventDefault();
                updateEraserSize(eraserSize + 5);
            }

            // + and - - Zoom
            if (e.key === '+' || e.key === '=') {
                e.preventDefault();
                setCleanZoom(cleanZoom + 0.25);
            }
            if (e.key === '-' || e.key === '_') {
                e.preventDefault();
                setCleanZoom(cleanZoom - 0.25);
            }

            // Arrow keys - Navigate
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                if (cleanIndex > 0) {
                    document.getElementById('prevCleanBtn').click();
                }
            }
            if (e.key === 'ArrowRight') {
                e.preventDefault();
                if (cleanIndex < cleanItems.length - 1) {
                    document.getElementById('nextCleanBtn').click();
                }
            }
        });

        document.getElementById('eraserSize').addEventListener('input', (e) => {
            updateEraserSize(parseInt(e.target.value));
        });

        // Straighten slider and button listeners
        const straightenSliderEl = document.getElementById('straightenSlider');
        if (straightenSliderEl) {
            straightenSliderEl.addEventListener('input', (e) => {
                const angle = parseFloat(e.target.value);
                applyStraightenAngle(angle);
            });

            straightenSliderEl.addEventListener('change', () => {
                undoStack.push(cleanCanvas.toDataURL());
                if (undoStack.length > 30) undoStack.shift();
                triggerCleanAutoSave();
                renderCleanThumbnails();
            });
        }

        const stepMinusBtn = document.getElementById('straightenStepMinusBtn');
        if (stepMinusBtn) stepMinusBtn.addEventListener('click', () => stepStraightenAngle(-0.1));

        const stepPlusBtn = document.getElementById('straightenStepPlusBtn');
        if (stepPlusBtn) stepPlusBtn.addEventListener('click', () => stepStraightenAngle(0.1));

        const fineMinusBtn = document.getElementById('straightenFineMinusBtn');
        if (fineMinusBtn) fineMinusBtn.addEventListener('click', () => stepStraightenAngle(-0.5));

        const finePlusBtn = document.getElementById('straightenFinePlusBtn');
        if (finePlusBtn) finePlusBtn.addEventListener('click', () => stepStraightenAngle(0.5));

        const straightenResetBtn = document.getElementById('straightenResetBtn');
        if (straightenResetBtn) {
            straightenResetBtn.addEventListener('click', () => {
                if (Math.abs(currentStraightenAngle) > 0.001) {
                    applyStraightenAngle(0);
                    undoStack.push(cleanCanvas.toDataURL());
                    if (undoStack.length > 30) undoStack.shift();
                    triggerCleanAutoSave();
                    renderCleanThumbnails();
                }
            });
        }

        const rotateCcw90Btn = document.getElementById('rotateCcw90Btn');
        if (rotateCcw90Btn) rotateCcw90Btn.addEventListener('click', () => rotateCleanCanvas90(-1));

        const rotateCw90Btn = document.getElementById('rotateCw90Btn');
        if (rotateCw90Btn) rotateCw90Btn.addEventListener('click', () => rotateCleanCanvas90(1));

        const toggleGridBtn = document.getElementById('toggleGridBtn');
        if (toggleGridBtn) toggleGridBtn.addEventListener('click', () => toggleCleanGrid());

        const trimBordersBtn = document.getElementById('trimBordersBtn');
        if (trimBordersBtn) trimBordersBtn.addEventListener('click', () => trimCanvasWhitespace(24));

        document.getElementById('undoEraseBtn').addEventListener('click', () => {
            undoClean();
        });

        document.getElementById('prevCleanBtn').addEventListener('click', async () => {
            if (cleanIndex > 0) {
                commitCleanRotation();
                await saveCurrentCleanDrawing();
                eraserMode = false;
                loadClean(cleanIndex - 1);
            }
        });

        document.getElementById('nextCleanBtn').addEventListener('click', async () => {
            if (cleanIndex < cleanItems.length - 1) {
                commitCleanRotation();
                await saveCurrentCleanDrawing();
                eraserMode = false;
                loadClean(cleanIndex + 1);
            }
        });

        document.getElementById('markCleanBtn').addEventListener('click', async () => {
            const item = cleanItems[cleanIndex];
            if (!item) return;

            commitCleanRotation();
            // Explicitly mark as cleaned (only the user action sets this)
            if (!state.cleanedDrawings[item.key]) state.cleanedDrawings[item.key] = {};
            state.cleanedDrawings[item.key].cleaned = true;

            await saveCurrentCleanDrawing();

            // Update thumbnails to show it's cleaned
            renderCleanThumbnails();

            // Auto-advance to next drawing
            if (cleanIndex < cleanItems.length - 1) {
                eraserMode = false;
                loadClean(cleanIndex + 1);
            }
        });


        document.getElementById('proceedReviewBtn').addEventListener('click', async () => {
            commitCleanRotation();
            await saveCurrentCleanDrawing();
            enableTab('review');
            switchTab('review');
            loadReviewTexts();
        });

        // TAB 5: Review Texts (2-COLUMN LAYOUT)
        function loadReviewTexts() {
            const container = document.getElementById('reviewContainer');
            container.innerHTML = '';

            console.log('[Review] Loading Review Texts...');
            console.log('   OCR Results:', Object.keys(state.ocrResults).length, 'keys');
            console.log('   Corrections:', Object.keys(state.corrections).length, 'keys');

            for (const imgName in state.annotations) {
                const ann = state.annotations[imgName];
                const imgElement = state.fullResImages[imgName];

                ann.drawings.forEach((drawing, dIdx) => {
                    const key = `${imgName}_d${dIdx + 1}`;
                    const ocrTexts = state.ocrResults[key] || [];

                        console.log(`   Drawing ${key}:`, ocrTexts.length, 'OCR texts');

                        drawing.textBoxes.forEach((textBox, tIdx) => {
                            const textKey = `${key}_t${tIdx + 1}`;

                            // Extract text box preview
                            const tempCanvas = document.createElement('canvas');
                            const tempCtx = tempCanvas.getContext('2d');
                            tempCanvas.width = textBox.w;
                            tempCanvas.height = textBox.h;
                            tempCtx.drawImage(imgElement, textBox.x, textBox.y, textBox.w, textBox.h, 0, 0, textBox.w, textBox.h);

                            const preview = tempCanvas.toDataURL();
                            const ocrText = ocrTexts[tIdx] || '';
                            const hasSavedCorrection = (state.corrections && (textKey in state.corrections) && state.corrections[textKey] !== ocrText);
                            const correctedText = hasSavedCorrection ? state.corrections[textKey] : ocrText;

                            console.log(`      Text ${textKey}: OCR="${ocrText}" | Corrected="${correctedText}"`);

                            const div = document.createElement('div');
                            div.className = 'text-box-review';
                            div.innerHTML = `
                                <div>
                                    <img src="${preview}" class="text-box-preview cursor-zoom-in" alt="Text ${tIdx + 1}" data-zoom-src="${preview}">
                                    <p class="text-xs text-gray-600 mt-2">
                                        <strong>${ann.metadata.tableName || imgName}</strong><br>
                                        D${dIdx + 1} - T${tIdx + 1}
                                    </p>
                                </div>
                                <div>
                                    <label class="block text-sm font-semibold mb-2">OCR Result (Editable):</label>
                                    <textarea class="ocr-correction-textarea w-full px-3 py-2 border-2 border-stone-300 rounded focus:border-[var(--primary)] font-mono text-sm" rows="4" data-key="${textKey}" data-original="${encodeURIComponent(ocrText)}">${correctedText}</textarea>
                                    <p class="mt-2 text-xs text-stone-500 italic"><i class="bi bi-pencil mr-1"></i> Changes are auto-saved</p>
                                </div>
                            `;
                            container.appendChild(div);
                        });
                    });
                }

                if (container.children.length === 0) {
                    container.innerHTML = '<p class="text-gray-500 text-center">No text boxes to review</p>';
                }

                // Add zoom functionality to all preview images
                document.querySelectorAll('.text-box-preview').forEach(img => {
                    img.addEventListener('click', () => {
                        const zoomModal = document.getElementById('imageZoomModal');
                        const zoomedImage = document.getElementById('zoomedImage');
                        zoomedImage.src = img.dataset.zoomSrc;
                        zoomModal.classList.remove('hidden');
                        document.body.classList.add('modal-open');
                    });
                });

                // Add auto-save functionality to all textareas (only counts genuine character edits)
                let saveTimeout = null;
                document.querySelectorAll('.ocr-correction-textarea').forEach(textarea => {
                    textarea.addEventListener('input', () => {
                        const textKey = textarea.dataset.key;
                        const originalValue = decodeURIComponent(textarea.dataset.original || '');
                        const newValue = textarea.value;

                        // Only record as a manual correction if user actually changed characters!
                        if (newValue !== originalValue) {
                            state.corrections[textKey] = newValue;
                        } else {
                            delete state.corrections[textKey];
                        }

                        clearTimeout(saveTimeout);
                        saveTimeout = setTimeout(async () => {
                            if (currentProject) {
                                console.log(`[AutoSave] Saving corrections... (${Object.keys(state.corrections).length} modified)`);
                                try {
                                    await fetch(`/api/project/${currentProject.project_id}/save_ocr_corrections`, {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ corrections: state.corrections })
                                    });
                                } catch (error) {
                                    console.error('[AutoSave] Save error:', error);
                                }
                            }
                        }, 800);
                    });
                });
            }

            // Close zoom modal handlers
            document.getElementById('closeZoomBtn').addEventListener('click', () => {
                document.getElementById('imageZoomModal').classList.add('hidden');
                document.body.classList.remove('modal-open');
            });

            document.getElementById('imageZoomModal').addEventListener('click', (e) => {
                if (e.target === document.getElementById('imageZoomModal')) {
                    document.getElementById('imageZoomModal').classList.add('hidden');
                    document.body.classList.remove('modal-open');
                }
            });

            document.getElementById('proceedExportBtn').addEventListener('click', async () => {
                // Update state only with actual differences
                document.querySelectorAll('.ocr-correction-textarea').forEach(textarea => {
                    const textKey = textarea.dataset.key;
                    const originalValue = decodeURIComponent(textarea.dataset.original || '');
                    const newValue = textarea.value;
                    if (newValue !== originalValue) {
                        state.corrections[textKey] = newValue;
                    } else {
                        delete state.corrections[textKey];
                    }
                });

                if (currentProject) {
                    console.log('[Export] Persisting verified OCR corrections before export...');
                    try {
                        await fetch(`/api/project/${currentProject.project_id}/save_ocr_corrections`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ corrections: state.corrections })
                        });
                    } catch (error) {
                        console.error('Error saving OCR corrections:', error);
                    }
                }

                enableTab('export');
                switchTab('export');
                updateExportSummary();
            });

            // TAB 6: Export Management & 1-Click Package Generation
            function countActualCorrections() {
                let count = 0;
                for (const imgName in state.annotations) {
                    const ann = state.annotations[imgName];
                    if (!ann || !ann.drawings) continue;
                    ann.drawings.forEach((drawing, dIdx) => {
                        const key = `${imgName}_d${dIdx + 1}`;
                        const ocrTexts = state.ocrResults[key] || [];
                        if (drawing.textBoxes) {
                            drawing.textBoxes.forEach((_, tIdx) => {
                                const textKey = `${key}_t${tIdx + 1}`;
                                const orig = ocrTexts[tIdx] || '';
                                const val = state.corrections[textKey];
                                if (val !== undefined && val !== orig) {
                                    count++;
                                }
                            });
                        }
                    });
                }
                return count;
            }

            function updateExportFilenamePreview() {
                const prefixInput = document.getElementById('exportPrefix');
                const previewEl = document.getElementById('exportFilenamePreview');
                if (!prefixInput || !previewEl) return;
                const rawVal = prefixInput.value.trim() || 'ceramic';
                const cleanPrefix = rawVal.replace(/[^a-zA-Z0-9_-]/g, '_');
                const timestamp = new Date().toISOString().split('T')[0];
                previewEl.textContent = `${cleanPrefix}_export_${timestamp}.zip`;
            }

            document.getElementById('exportPrefix')?.addEventListener('input', updateExportFilenamePreview);

            function updateExportSummary() {
                const totalPlates = state.images.length;
                let totalDrawings = 0;
                let totalOcrBoxes = 0;
                for (const ann of Object.values(state.annotations)) {
                    if (ann && ann.drawings) {
                        totalDrawings += ann.drawings.length;
                        ann.drawings.forEach(d => {
                            if (d.textBoxes) totalOcrBoxes += d.textBoxes.length;
                        });
                    }
                }

                const correctionsCount = countActualCorrections();

                const platesEl = document.getElementById('exportTotalPlates');
                const drawingsEl = document.getElementById('exportTotalDrawings');
                const ocrEl = document.getElementById('exportOcrCount');
                const correctionsEl = document.getElementById('exportCorrectionCount');

                if (platesEl) platesEl.textContent = totalPlates;
                if (drawingsEl) drawingsEl.textContent = totalDrawings;
                if (ocrEl) ocrEl.textContent = totalOcrBoxes;
                if (correctionsEl) correctionsEl.textContent = correctionsCount;

                updateExportFilenamePreview();
            }

            // Compile export dataset in memory
            async function compileExportPayload(prefix, onProgress) {
                const dataRows = [];
                const mlDataRows = [];
                const images = [];
                let imageIndex = 0;

                let totalDrawings = 0;
                for (const imgName in state.annotations) {
                    const ann = state.annotations[imgName];
                    if (ann && ann.drawings) totalDrawings += ann.drawings.length;
                }

                for (const imgName in state.annotations) {
                    const ann = state.annotations[imgName];
                    const imgElement = state.fullResImages[imgName];
                    if (!ann || !ann.drawings || !imgElement) continue;

                    for (let dIdx = 0; dIdx < ann.drawings.length; dIdx++) {
                        const drawing = ann.drawings[dIdx];
                        const key = `${imgName}_d${dIdx + 1}`;

                        // Render drawing profile (cleaned canvas if available, or full-res crop)
                        const tempCanvas = document.createElement('canvas');
                        const tempCtx = tempCanvas.getContext('2d');

                        const cleanedData = state.cleanedDrawings[key];
                        if (cleanedData && cleanedData.imageData) {
                            const cleanImg = new Image();
                            await new Promise((resolve) => {
                                cleanImg.onload = () => {
                                    tempCanvas.width = cleanImg.naturalWidth || cleanImg.width;
                                    tempCanvas.height = cleanImg.naturalHeight || cleanImg.height;
                                    tempCtx.fillStyle = '#ffffff';
                                    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                                    tempCtx.drawImage(cleanImg, 0, 0);
                                    resolve();
                                };
                                cleanImg.onerror = resolve;
                                cleanImg.src = cleanedData.imageData;
                            });
                        } else {
                            tempCanvas.width = drawing.w;
                            tempCanvas.height = drawing.h;
                            tempCtx.fillStyle = '#ffffff';
                            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                            tempCtx.drawImage(imgElement, drawing.x, drawing.y, drawing.w, drawing.h, 0, 0, drawing.w, drawing.h);
                        }

                        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.95);
                        const filename = `${prefix}_${String(imageIndex + 1).padStart(3, '0')}.jpg`;

                        images.push({
                            filename: filename,
                            data: dataUrl
                        });

                        // Collect OCR and manual corrections (cleaned of stray line breaks for pristine CSV and Excel rows)
                        const ocrTexts = (state.ocrResults[key] || []).map(t => (t || '').replace(/\r?\n+/g, ' ').trim());
                        const ocrResult = ocrTexts.join(' | ');

                        const correctedTexts = [];
                        if (drawing.textBoxes) {
                            drawing.textBoxes.forEach((textBox, tIdx) => {
                                const textKey = `${key}_t${tIdx + 1}`;
                                const rawVal = state.corrections[textKey] !== undefined ? state.corrections[textKey] : (ocrTexts[tIdx] || '');
                                correctedTexts.push((rawVal || '').replace(/\r?\n+/g, ' ').trim());
                            });
                        }
                        const correctedResult = correctedTexts.join(' | ');

                        dataRows.push({
                            filename: filename,
                            original_image: imgName,
                            drawing_number: dIdx + 1,
                            table_name: ann.metadata.tableName || '',
                            context: ann.metadata.context || '',
                            notes: ann.metadata.notes || '',
                            ocr_result: ocrResult,
                            ocr_corrected: correctedResult
                        });

                        // ML Ground Truth bounding boxes
                        mlDataRows.push({
                            image_name: imgName,
                            drawing_number: dIdx + 1,
                            box_type: 'vessel',
                            x: drawing.x,
                            y: drawing.y,
                            width: drawing.w,
                            height: drawing.h,
                            label: drawing.label || 'vessel',
                            text_index: '',
                            image_width: imgElement.naturalWidth,
                            image_height: imgElement.naturalHeight
                        });

                        if (drawing.textBoxes) {
                            drawing.textBoxes.forEach((textBox, tIdx) => {
                                mlDataRows.push({
                                    image_name: imgName,
                                    drawing_number: dIdx + 1,
                                    box_type: 'text',
                                    x: textBox.x,
                                    y: textBox.y,
                                    width: textBox.w,
                                    height: textBox.h,
                                    label: 'text_label',
                                    text_index: tIdx + 1,
                                    image_width: imgElement.naturalWidth,
                                    image_height: imgElement.naturalHeight
                                });
                            });
                        }

                        imageIndex++;
                        if (onProgress) onProgress(imageIndex, totalDrawings);
                    }
                }

                // Generate XLSX files in base64 if SheetJS is loaded
                let catalogXlsxBase64 = null;
                let mlXlsxBase64 = null;

                if (typeof XLSX !== 'undefined') {
                    try {
                        const ws = XLSX.utils.json_to_sheet(dataRows);
                        const wb = XLSX.utils.book_new();
                        XLSX.utils.book_append_sheet(wb, ws, 'Catalogue & OCR');
                        const xlsxBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'base64' });
                        catalogXlsxBase64 = xlsxBuffer;

                        const mlWs = XLSX.utils.json_to_sheet(mlDataRows);
                        const mlWb = XLSX.utils.book_new();
                        XLSX.utils.book_append_sheet(mlWb, mlWs, 'Bounding Boxes');
                        const mlXlsxBuffer = XLSX.write(mlWb, { bookType: 'xlsx', type: 'base64' });
                        mlXlsxBase64 = mlXlsxBuffer;
                    } catch (xErr) {
                        console.warn('Could not generate client XLSX base64:', xErr);
                    }
                }

                return {
                    prefix: prefix,
                    catalog_rows: dataRows,
                    ml_rows: mlDataRows,
                    images: images,
                    catalog_xlsx_b64: catalogXlsxBase64,
                    ml_xlsx_b64: mlXlsxBase64
                };
            }

            // 1-Click Complete ZIP Export
            document.getElementById('downloadZipBtn')?.addEventListener('click', async () => {
                const btn = document.getElementById('downloadZipBtn');
                const progressStrip = document.getElementById('exportProgressStrip');
                const progressMsg = document.getElementById('exportProgressMsg');
                const progressBarInner = document.getElementById('exportProgressBarInner');
                const successRow = document.getElementById('exportSuccessNextRow');

                const prefixInput = document.getElementById('exportPrefix');
                const cleanPrefix = (prefixInput?.value.trim() || 'ceramic').replace(/[^a-zA-Z0-9_-]/g, '_');
                const timestamp = new Date().toISOString().split('T')[0];

                try {
                    btn.disabled = true;
                    if (progressStrip) progressStrip.classList.remove('hidden');
                    if (progressMsg) progressMsg.textContent = 'Processing high-resolution vessel drawings...';
                    if (progressBarInner) progressBarInner.style.width = '20%';

                    const payload = await compileExportPayload(cleanPrefix, (current, total) => {
                        if (progressMsg) progressMsg.textContent = `Rendering drawing ${current} of ${total}...`;
                        if (progressBarInner) progressBarInner.style.width = `${Math.round((current / total) * 60)}%`;
                    });

                    if (progressMsg) progressMsg.textContent = 'Packaging ZIP archive on server...';
                    if (progressBarInner) progressBarInner.style.width = '80%';

                    const projectId = currentProject ? currentProject.project_id : 'default';
                    const response = await fetch(`/api/project/${projectId}/export_zip`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (!response.ok) {
                        let errMsg = `Server returned HTTP ${response.status}`;
                        try {
                            const errJson = await response.json();
                            if (errJson && errJson.error) errMsg = errJson.error;
                        } catch (e) {}
                        throw new Error(errMsg);
                    }

                    if (progressMsg) progressMsg.textContent = 'Downloading ZIP archive...';
                    if (progressBarInner) progressBarInner.style.width = '98%';

                    const blob = await response.blob();
                    const downloadUrl = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = downloadUrl;
                    a.download = `${cleanPrefix}_export_${timestamp}.zip`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(downloadUrl);

                    if (progressBarInner) progressBarInner.style.width = '100%';
                    setTimeout(() => {
                        if (progressStrip) progressStrip.classList.add('hidden');
                    }, 600);

                    if (successRow) successRow.classList.remove('hidden');
                    enableTab('parser');

                } catch (err) {
                    console.error('ZIP Export error:', err);
                    alert('Error during export: ' + err.message);
                    if (progressStrip) progressStrip.classList.add('hidden');
                } finally {
                    btn.disabled = false;
                }
            });

            // 1-Click Excel Only Download
            document.getElementById('downloadXlsxBtn')?.addEventListener('click', async () => {
                const btn = document.getElementById('downloadXlsxBtn');
                const prefixInput = document.getElementById('exportPrefix');
                const cleanPrefix = (prefixInput?.value.trim() || 'ceramic').replace(/[^a-zA-Z0-9_-]/g, '_');
                const timestamp = new Date().toISOString().split('T')[0];

                try {
                    btn.disabled = true;
                    btn.innerHTML = '<div class="ocr-spinner" style="width: 16px; height: 16px; border-width: 2px; margin: 0;"></div> Generating Excel...';

                    const payload = await compileExportPayload(cleanPrefix);
                    const projectId = currentProject ? currentProject.project_id : 'default';

                    // 1. Try downloading from backend /api/project/<id>/export_excel (full formatting & column widths)
                    try {
                        const response = await fetch(`/api/project/${projectId}/export_excel`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        if (response.ok) {
                            const blob = await response.blob();
                            const downloadUrl = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = downloadUrl;
                            a.download = `${cleanPrefix}_metadata_${timestamp}.xlsx`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(downloadUrl);

                            const successRow = document.getElementById('exportSuccessNextRow');
                            if (successRow) successRow.classList.remove('hidden');
                            enableTab('parser');
                            return;
                        }
                    } catch (netErr) {
                        console.warn('Server Excel generation failed, trying client SheetJS:', netErr);
                    }

                    // 2. Client SheetJS fallback
                    if (typeof XLSX !== 'undefined') {
                        const ws = XLSX.utils.json_to_sheet(payload.catalog_rows);
                        const wb = XLSX.utils.book_new();
                        XLSX.utils.book_append_sheet(wb, ws, 'Catalogue & OCR');

                        const mlWs = XLSX.utils.json_to_sheet(payload.ml_rows);
                        XLSX.utils.book_append_sheet(wb, mlWs, 'ML Bounding Boxes');

                        XLSX.writeFile(wb, `${cleanPrefix}_metadata_${timestamp}.xlsx`);
                    } else {
                        // 3. Standard RFC-4180 CSV fallback with UTF-8 BOM
                        const csvRows = payload.catalog_rows;
                        if (csvRows.length > 0) {
                            const keys = Object.keys(csvRows[0]);
                            let csvContent = '\ufeff' + keys.join(',') + '\r\n';
                            csvRows.forEach(row => {
                                csvContent += keys.map(k => {
                                    const val = String(row[k] ?? '').replace(/\r?\n+/g, ' ').replace(/"/g, '""');
                                    return `"${val}"`;
                                }).join(',') + '\r\n';
                            });
                            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `${cleanPrefix}_metadata_${timestamp}.csv`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                        }
                    }

                    const successRow = document.getElementById('exportSuccessNextRow');
                    if (successRow) successRow.classList.remove('hidden');
                    enableTab('parser');

                } catch (err) {
                    console.error('Excel Export error:', err);
                    alert('Error generating Excel: ' + err.message);
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="bi bi-file-earmark-excel text-emerald-700 text-base"></i> Download Excel (.xlsx)';
                }
            });

        document.getElementById('proceedParserBtn').addEventListener('click', () => {
            switchTab('parser');
        });

        // ======================
        // FEW-SHOT EXAMPLES PERSISTENCE
        // ======================

        async function saveFewShotExamplesToProject(projectId) {
            if (!projectId) return;

            try {
                console.log(`[Parser] Saving ${parserState.fewshot.length} few-shot examples to project...`);

                const response = await fetch(`/api/project/${projectId}/save_fewshot_examples`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ examples: parserState.fewshot })
                });

                const data = await response.json();

                if (data.success) {
                    console.log(`[Parser] Few-shot examples saved: ${data.count} examples`);
                } else {
                    console.error('[Parser] Failed to save few-shot examples:', data.error);
                }
            } catch (error) {
                console.error('[Parser] Error saving few-shot examples:', error);
            }
        }

        async function loadFewShotExamplesFromProject(projectId) {
            if (!projectId) return;

            try {
                console.log('[Parser] Loading few-shot examples from project...');

                const response = await fetch(`/api/project/${projectId}/fewshot_examples`);
                const data = await response.json();

                if (data.success && data.examples && data.examples.length > 0) {
                    console.log(`[Parser] Loaded ${data.count} few-shot examples from project`);
                    parserState.fewshot = data.examples;
                    renderExamples();
                    console.log('[Parser] Few-shot examples restored in UI');
                } else {
                    parserState.fewshot = [];
                    renderExamples();
                    console.log('[Parser] No few-shot examples found in project');
                }
            } catch (error) {
                console.error('[Parser] Error loading few-shot examples:', error);
            }
        }

        // ======================
        // TAB 7: FEW-SHOT PARSER
        // ======================
        const parserState = {
            fewshot: [],
            ocrLines: [],
            filenames: [],  // Store corresponding filenames
            currentIndex: 0,
            currentJson: {},
            originalText: '',
            dirty: false,  // fields assigned on the current line that were not added as an example yet
            allFields: [
                { name: 'Inventory', key: 'inventario', fixed: true },
                { name: 'Site', key: 'sito', fixed: true },
                { name: 'Year', key: 'anno', fixed: true },
                { name: 'US', key: 'us', fixed: true },
                { name: 'Area', key: 'area', fixed: true },
                { name: 'Cut', key: 'taglio', fixed: true },
                { name: 'Sector', key: 'settore', fixed: true },
                { name: 'Notes', key: 'note', fixed: true }
            ]
        };

        const highlightColors = ['#ffff99', '#99ff99', '#99ccff', '#ffcc99', '#ff99cc', '#cc99ff', '#99ffff', '#ffffcc', '#ccff99', '#99ffcc'];

        function capitalize(str) {
            return str.charAt(0).toUpperCase() + str.slice(1);
        }

        function escapeHtml(str) {
            if (!str) return '';
            return String(str)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function renderTags() {
            const tagsList = document.getElementById('tags-list');
            if (!tagsList) return;
            tagsList.innerHTML = '';
            parserState.allFields.forEach((field, index) => {
                const tagDiv = document.createElement('div');
                tagDiv.className = 'parser-field-item';
                const dotColor = highlightColors[index % highlightColors.length];
                tagDiv.innerHTML = `
                    <span class="flex items-center gap-2">
                        <span class="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style="background-color: ${dotColor};"></span>
                        <span class="font-semibold text-xs text-stone-800">${escapeHtml(field.name)}</span>
                    </span>
                    ${!field.fixed ? `<button class="delete-tag-btn" data-key="${escapeHtml(field.key)}" title="Delete field"><i class="bi bi-trash3 pointer-events-none"></i></button>` : ''}
                `;
                tagsList.appendChild(tagDiv);
            });
        }

        function renderPopupTags() {
            const popupTags = document.getElementById('popup-tags');
            popupTags.innerHTML = '';
            parserState.allFields.forEach(field => {
                const btn = document.createElement('button');
                btn.className = 'popup-tag-btn';
                btn.textContent = field.name;
                btn.dataset.field = field.key;
                popupTags.appendChild(btn);
            });
        }

        function updateJsonPreview() {
            const tableBody = document.getElementById('json-preview-table');
            tableBody.innerHTML = '';

            // Create table rows for each field
            Object.entries(parserState.currentJson).forEach(([field, value]) => {
                const row = document.createElement('tr');
                row.className = 'hover:bg-gray-50 transition-colors';

                const hasValue = value !== null && value !== '';
                const fieldCell = document.createElement('td');
                fieldCell.className = 'px-4 py-3 text-sm font-medium text-gray-900';
                fieldCell.innerHTML = `
                    <span class="inline-flex items-center gap-2">
                        ${hasValue ? '<span class="text-teal-600 text-sm"><i class="bi bi-check-circle-fill"></i></span>' : '<span class="text-stone-300 text-sm"><i class="bi bi-circle"></i></span>'}
                        ${field}
                    </span>
                `;

                const valueCell = document.createElement('td');
                valueCell.className = 'px-4 py-3 text-sm';

                if (hasValue) {
                    valueCell.innerHTML = `<span class="text-gray-900 font-medium">${value}</span>`;
                } else {
                    valueCell.innerHTML = '<span class="text-gray-400 italic">not assigned</span>';
                }

                const actionCell = document.createElement('td');
                actionCell.className = 'px-2 py-3 text-right';
                if (hasValue) {
                    const removeBtn = document.createElement('button');
                    removeBtn.type = 'button';
                    removeBtn.className = 'delete-example-btn remove-field-btn';
                    removeBtn.dataset.field = field;
                    removeBtn.title = 'Remove this assignment';
                    removeBtn.setAttribute('aria-label', `Remove ${field}`);
                    removeBtn.innerHTML = '<i class="bi bi-x-lg"></i>';
                    actionCell.appendChild(removeBtn);
                }

                row.appendChild(fieldCell);
                row.appendChild(valueCell);
                row.appendChild(actionCell);
                tableBody.appendChild(row);
            });
        }

        // Remove the highlight of a field from the OCR text (keeps the text itself)
        function unwrapFieldHighlight(fieldKey, keepSpan = null) {
            const container = document.getElementById('ocr-text');
            container.querySelectorAll('span.' + CSS.escape(fieldKey)).forEach(span => {
                if (span !== keepSpan) span.replaceWith(...span.childNodes);
            });
            container.normalize();
        }

        function refreshDirty() {
            parserState.dirty = Object.values(parserState.currentJson).some(v => v !== null && v !== '');
        }

        // Ask before leaving a line whose assigned fields were not added as an example
        async function confirmLeaveUnsaved(message) {
            if (!parserState.dirty) return true;
            return await showConfirmDialog({
                title: 'Unsaved parsing',
                message,
                confirmText: 'Continue without saving',
                cancelText: 'Stay',
                type: 'warning'
            });
        }

        function loadCurrentLine() {
            parserState.originalText = parserState.ocrLines[parserState.currentIndex];
            document.getElementById('ocr-text').innerHTML = parserState.originalText;
            parserState.currentJson = {};
            parserState.allFields.forEach(field => {
                parserState.currentJson[capitalize(field.name)] = null;
            });
            parserState.dirty = false;
            updateJsonPreview();
            document.getElementById('current-index').textContent = `${parserState.currentIndex + 1} / ${parserState.ocrLines.length}`;
        }

        // Add tag button
        document.getElementById('add-tag-btn').addEventListener('click', async () => {
            const fieldName = await showPromptDialog({
                title: 'Add New Field',
                message: 'Enter the label for the new semantic field to extract:',
                placeholder: 'e.g. Chronology, Fabric, Form',
                confirmText: 'Add Field',
                cancelText: 'Cancel'
            });

            if (fieldName && fieldName.trim()) {
                const fieldKey = fieldName.toLowerCase().replace(/\s+/g, '_');
                if (parserState.allFields.some(f => f.key === fieldKey)) {
                    await showAlertDialog({
                        title: 'Field Exists',
                        message: `A field with key "${fieldKey}" already exists!`,
                        type: 'warning'
                    });
                    return;
                }
                parserState.allFields.push({ name: fieldName.trim(), key: fieldKey, fixed: false });
                renderTags();
                renderPopupTags();

                // Add highlight style
                const styleEl = document.createElement('style');
                styleEl.textContent = `.${fieldKey} { background-color: ${highlightColors[(parserState.allFields.length - 1) % highlightColors.length]}; }`;
                document.head.appendChild(styleEl);

                parserState.currentJson[capitalize(fieldName.trim())] = null;
                updateJsonPreview();
            }
        });

        // Delete tag
        document.getElementById('tags-list').addEventListener('click', (e) => {
            if (e.target.classList.contains('delete-tag-btn')) {
                const key = e.target.dataset.key;
                const field = parserState.allFields.find(f => f.key === key);
                if (field && !field.fixed) {
                    parserState.allFields = parserState.allFields.filter(f => f.key !== key);
                    renderTags();
                    renderPopupTags();
                    delete parserState.currentJson[capitalize(field.name)];
                    updateJsonPreview();
                }
            }
        });

        // Trigger custom file picker button
        const triggerCsvBtn = document.getElementById('triggerCsvUploadBtn');
        if (triggerCsvBtn) {
            triggerCsvBtn.addEventListener('click', () => {
                const fileInput = document.getElementById('csv-file');
                if (fileInput) fileInput.click();
            });
        }

        // CSV/Excel file upload
        document.getElementById('csv-file').addEventListener('change', (e) => {
            const file = e.target.files[0];
            const nameBadge = document.getElementById('selectedFileNameBadge');
            const nameText = document.getElementById('selectedFileNameText');

            if (file) {
                if (nameText) nameText.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                if (nameBadge) nameBadge.classList.add('has-file');

                const reader = new FileReader();
                reader.onload = (event) => {
                    const data = new Uint8Array(event.target.result);
                    const workbook = XLSX.read(data, { type: 'array' });

                    // Get first sheet
                    const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
                    const jsonData = XLSX.utils.sheet_to_json(firstSheet, { header: 1 });

                    if (jsonData.length < 2) {
                        alert('Excel file is empty or has no data rows.');
                        return;
                    }

                    const headers = jsonData[0];
                    const rows = jsonData.slice(1);

                    // Find column indices
                    const ocrColIndex = headers.findIndex(h => h && h.toString().includes('ocr_corrected'));
                    const filenameColIndex = headers.findIndex(h => h && h.toString().includes('filename'));

                    if (ocrColIndex === -1) {
                        alert('Column "ocr_corrected" not found in Excel file.');
                        return;
                    }

                    // Parse each row and store both OCR text and filename
                    parserState.ocrLines = [];
                    parserState.filenames = [];

                    rows.forEach(row => {
                        const ocrText = row[ocrColIndex] ? row[ocrColIndex].toString().trim() : '';
                        const filename = filenameColIndex >= 0 && row[filenameColIndex] ?
                            row[filenameColIndex].toString().trim() : '';

                        if (ocrText) {
                            parserState.ocrLines.push(ocrText);
                            parserState.filenames.push(filename);
                        }
                    });

                    if (parserState.ocrLines.length > 0) {
                        document.getElementById('navigation').classList.remove('hidden');
                        document.getElementById('add-example').classList.remove('hidden');
                        parserState.currentIndex = 0;
                        loadCurrentLine();
                    }
                };
                reader.readAsArrayBuffer(file);
            }
        });

        // Text selection for highlighting
        document.getElementById('ocr-text').addEventListener('mouseup', (e) => {
            const selection = window.getSelection();
            const selectionPopup = document.getElementById('selection-popup');

            const selectedText = selection.toString().trim();
            if (selection.rangeCount > 0 && selectedText) {
                const range = selection.getRangeAt(0);
                const rect = range.getBoundingClientRect();

                // Reveal popup so offsetWidth/offsetHeight can be accurately measured
                selectionPopup.classList.remove('hidden');
                selectionPopup.style.visibility = 'hidden';

                const popupWidth = selectionPopup.offsetWidth || 200;
                const popupHeight = selectionPopup.offsetHeight || 220;

                const selCenterX = rect.left + (rect.width / 2);
                let left = Math.round(selCenterX - (popupWidth / 2));

                const viewportWidth = window.innerWidth;
                const viewportHeight = window.innerHeight;

                // Keep popup within viewport horizontally
                left = Math.max(10, Math.min(viewportWidth - popupWidth - 10, left));

                // Position arrow tip relative to popup
                const arrowLeft = Math.max(16, Math.min(popupWidth - 16, Math.round(selCenterX - left)));
                selectionPopup.style.setProperty('--arrow-left', `${arrowLeft}px`);

                // 9px arrow height + 2px gap = 11px
                const arrowGap = 11;

                // Prefer positioning above selection
                let top = Math.round(rect.top - popupHeight - arrowGap);
                let placeAbove = true;

                // If not enough space above, position below
                if (top < 10) {
                    top = Math.round(rect.bottom + arrowGap);
                    placeAbove = false;
                    if (top + popupHeight > viewportHeight - 10) {
                        top = viewportHeight - popupHeight - 10;
                    }
                }

                if (placeAbove) {
                    selectionPopup.classList.add('arrow-bottom');
                    selectionPopup.classList.remove('arrow-top');
                } else {
                    selectionPopup.classList.add('arrow-top');
                    selectionPopup.classList.remove('arrow-bottom');
                }

                selectionPopup.style.left = `${left}px`;
                selectionPopup.style.top = `${top}px`;
                selectionPopup.style.width = `${popupWidth}px`;
                selectionPopup.style.visibility = 'visible';

            } else {
                selectionPopup.classList.add('hidden');
            }
        });

        // Assign field to selection
        document.getElementById('popup-tags').addEventListener('click', (e) => {
            if (e.target.classList.contains('popup-tag-btn')) {
                const selection = window.getSelection();
                const selectionPopup = document.getElementById('selection-popup');

                if (selection.rangeCount > 0) {
                    const range = selection.getRangeAt(0);
                    const fieldKey = e.target.dataset.field;
                    const field = parserState.allFields.find(f => f.key === fieldKey);
                    if (field) {
                        const value = range.toString().trim();
                        parserState.currentJson[capitalize(field.name)] = value;

                        // Highlight the selected text
                        const span = document.createElement('span');
                        span.className = fieldKey;
                        try {
                            range.surroundContents(span);
                        } catch (err) {
                            // If surroundContents fails (e.g., partial node selection), 
                            // use a different approach
                            const fragment = range.extractContents();
                            span.appendChild(fragment);
                            range.insertNode(span);
                        }

                        // Assigning a field again replaces the previous one: drop its old highlight
                        unwrapFieldHighlight(fieldKey, span);

                        refreshDirty();
                        updateJsonPreview();
                        selection.removeAllRanges();
                        selectionPopup.classList.add('hidden');
                    }
                }
            }
        });

        // Close popup on outside click
        document.addEventListener('click', (e) => {
            const selectionPopup = document.getElementById('selection-popup');
            const ocrText = document.getElementById('ocr-text');
            if (!selectionPopup.contains(e.target) && !ocrText.contains(e.target)) {
                selectionPopup.classList.add('hidden');
            }
        });

        // Navigation buttons
        const UNSAVED_LINE_MESSAGE = 'You assigned fields on this line but have not added it as a few-shot example. Click "Add Few-Shot Example" to keep it. Continue without saving?';

        document.getElementById('prev-btn').addEventListener('click', async () => {
            if (parserState.currentIndex > 0) {
                if (!await confirmLeaveUnsaved(UNSAVED_LINE_MESSAGE)) return;
                parserState.currentIndex--;
                loadCurrentLine();
                updateJsonPreview();  // Update table when navigating
            }
        });

        document.getElementById('next-btn').addEventListener('click', async () => {
            if (parserState.currentIndex < parserState.ocrLines.length - 1) {
                if (!await confirmLeaveUnsaved(UNSAVED_LINE_MESSAGE)) return;
                parserState.currentIndex++;
                loadCurrentLine();
                updateJsonPreview();  // Update table when navigating
            }
        });

        // Remove a single assigned field from the current line
        document.getElementById('json-preview-table').addEventListener('click', (e) => {
            const btn = e.target.closest('.remove-field-btn');
            if (!btn) return;
            const fieldName = btn.dataset.field;
            const field = parserState.allFields.find(f => capitalize(f.name) === fieldName);
            parserState.currentJson[fieldName] = null;
            if (field) unwrapFieldHighlight(field.key);
            refreshDirty();
            updateJsonPreview();
        });

        document.getElementById('reset-btn').addEventListener('click', async () => {
            if (!await confirmLeaveUnsaved('Reset this line? The fields you assigned and did not add as a few-shot example will be cleared.')) return;
            loadCurrentLine();
            updateJsonPreview();  // Update table when resetting
        });

        // Add example
        document.getElementById('add-example').addEventListener('click', async () => {
            if (parserState.originalText) {
                parserState.fewshot.push({ role: "user", content: parserState.originalText });
                parserState.fewshot.push({ role: "assistant", content: JSON.stringify(parserState.currentJson) });

                parserState.dirty = false;
                renderExamples();

                // Auto-save to project if a project is open
                if (currentProject) {
                    await saveFewShotExamplesToProject(currentProject.project_id);
                }

                if (parserState.currentIndex < parserState.ocrLines.length - 1) {
                    parserState.currentIndex++;
                    loadCurrentLine();
                }
            }
        });

        // The examples list is collapsed by default; the header toggles it (the count stays visible)
        document.getElementById('toggleExamplesBtn').addEventListener('click', () => {
            const list = document.getElementById('examples-container');
            const btn = document.getElementById('toggleExamplesBtn');
            const isOpen = list.classList.toggle('hidden') === false;
            btn.setAttribute('aria-expanded', String(isOpen));
        });

        // Clear all examples
        document.getElementById('clear-examples-btn').addEventListener('click', async () => {
            const confirmed = await showConfirmDialog({
                title: 'Clear Examples',
                message: 'Are you sure you want to clear all few-shot examples? This cannot be undone.',
                confirmText: 'Clear All',
                cancelText: 'Cancel',
                type: 'danger',
                icon: 'bi-trash3-fill'
            });

            if (!confirmed) return;

            // Clear from memory
            parserState.fewshot = [];

            // Re-render UI
            renderExamples();

            // Save empty list to project if a project is open
            if (currentProject) {
                await saveFewShotExamplesToProject(currentProject.project_id);
                console.log('[Parser] Few-shot examples cleared from project');
            }

            console.log('[Parser] All few-shot examples cleared');
        });

        // Delete single example - use event delegation
        document.getElementById('examples-container').addEventListener('click', async (e) => {
            const btn = e.target.closest('.delete-example-btn');
            if (btn) {
                const exampleIndex = parseInt(btn.dataset.index);

                const confirmed = await showConfirmDialog({
                    title: 'Delete Example',
                    message: `Delete few-shot example ${exampleIndex + 1}?`,
                    confirmText: 'Delete',
                    cancelText: 'Cancel',
                    type: 'danger',
                    icon: 'bi-trash3'
                });

                if (!confirmed) return;

                // Remove from fewshot array (each example is 2 items: user + assistant)
                const arrayIndex = exampleIndex * 2;
                parserState.fewshot.splice(arrayIndex, 2);

                // Re-render all examples to update indices
                renderExamples();

                // Save to project if a project is open
                if (currentProject) {
                    await saveFewShotExamplesToProject(currentProject.project_id);
                    console.log(`[Parser] Example ${exampleIndex + 1} deleted and saved to project`);
                } else {
                    console.log(`[Parser] Example ${exampleIndex + 1} deleted`);
                }
            }
        });

        // Helper function to generate table HTML from JSON
        function generateJsonTable(jsonObj) {
            let tableHtml = '<div class="parser-table-container mt-1.5">';
            tableHtml += '<table class="parser-table">';
            tableHtml += '<thead><tr>';
            tableHtml += '<th style="width: 35%;">Field</th>';
            tableHtml += '<th>Value</th>';
            tableHtml += '</tr></thead><tbody>';

            Object.entries(jsonObj).forEach(([field, value]) => {
                const hasValue = value !== null && value !== undefined && value !== '';
                const displayValue = hasValue ? escapeHtml(String(value)) : 'not assigned';
                const indicator = hasValue ? '<span class="text-teal-600 text-xs mr-1.5"><i class="bi bi-check-circle-fill"></i></span>' : '<span class="text-stone-300 text-xs mr-1.5"><i class="bi bi-circle"></i></span>';
                const valueClass = hasValue ? 'text-stone-900 font-medium' : 'text-stone-400 italic';

                tableHtml += '<tr>';
                tableHtml += `<td class="whitespace-nowrap font-medium text-stone-700">${indicator}${escapeHtml(field)}</td>`;
                tableHtml += `<td><span class="${valueClass}">${displayValue}</span></td>`;
                tableHtml += '</tr>';
            });

            tableHtml += '</tbody></table></div>';
            return tableHtml;
        }

        // Render examples in DOM
        function renderExamples() {
            const container = document.getElementById('examples-container');
            const badge = document.getElementById('examplesCountBadge');
            const totalCount = Math.floor(parserState.fewshot.length / 2);

            if (badge) {
                badge.textContent = totalCount;
            }

            if (!container) return;
            container.innerHTML = '';

            if (totalCount === 0) {
                container.innerHTML = '<div class="parser-empty-state"><i class="bi bi-inbox mr-1"></i> No few-shot examples added yet.</div>';
                return;
            }

            for (let i = 0; i < parserState.fewshot.length; i += 2) {
                const userMsg = parserState.fewshot[i];
                const assistantMsg = parserState.fewshot[i + 1];
                const exampleIndex = i / 2;

                const exampleDiv = document.createElement('div');
                exampleDiv.className = 'parser-example-card';
                exampleDiv.dataset.exampleIndex = exampleIndex;

                let jsonObj;
                try {
                    jsonObj = JSON.parse(assistantMsg.content);
                } catch (e) {
                    jsonObj = {};
                }
                const tableHtml = generateJsonTable(jsonObj);
                const safeUserContent = escapeHtml(userMsg.content);

                exampleDiv.innerHTML = `
                    <div class="parser-example-header">
                        <span class="parser-example-title">
                            <i class="bi bi-bookmark-check-fill" style="color: var(--primary);"></i>
                            Example ${exampleIndex + 1}
                        </span>
                        <button class="delete-example-btn" data-index="${exampleIndex}" title="Delete this example" type="button">
                            <i class="bi bi-trash3 pointer-events-none"></i>
                        </button>
                    </div>
                    <div class="parser-example-body">
                        <div class="parser-example-field-group">
                            <span class="parser-example-label">
                                <i class="bi bi-file-earmark-text" style="color: var(--primary);"></i> OCR Text:
                            </span>
                            <div class="parser-example-ocr-text">${safeUserContent}</div>
                        </div>
                        <div class="parser-example-field-group">
                            <span class="parser-example-label">
                                <i class="bi bi-table" style="color: var(--teal);"></i> Parsed Data:
                            </span>
                            ${tableHtml}
                        </div>
                    </div>
                `;
                container.appendChild(exampleDiv);
            }
        }
        window.renderAllExamples = renderExamples;

        // Run parsing
        document.getElementById('run-parsing-btn').addEventListener('click', async () => {
            if (window.qwenAvailable === false) {
                alert('The parsing model (Qwen3.5-2B) could not be downloaded on this system, so structured parsing is unavailable. Check your internet connection and restart PyPotteryScan to retry the download.');
                return;
            }

            if (parserState.fewshot.length === 0) {
                alert('No few-shot examples created! Add at least one example first.');
                return;
            }

            const btn = document.getElementById('run-parsing-btn');
            const logDiv = document.getElementById('parsing-log');
            const logContent = document.getElementById('parsing-log-content');

            btn.disabled = true;
            btn.innerHTML = '<i class="bi bi-arrow-repeat spin mr-1"></i> Processing...';
            logDiv.classList.remove('hidden');
            logContent.innerHTML = '<p style="color: var(--primary);"><i class="bi bi-search mr-1"></i> Starting structured parsing...</p>';

            try {
                logContent.innerHTML += `<p><i class="bi bi-card-list mr-1"></i> Total lines to parse: ${parserState.ocrLines.length}</p>`;
                logContent.innerHTML += `<p><i class="bi bi-bullseye mr-1"></i> Using ${parserState.fewshot.length / 2} few-shot examples</p>`;
                logContent.innerHTML += '<p style="color: #b45309;"><i class="bi bi-box-seam mr-1"></i> Loading Qwen model...</p>';

                // Prepare data for parsing
                const payload = {
                    ocrLines: parserState.ocrLines,
                    filenames: parserState.filenames,  // Include filenames
                    fewshotExamples: parserState.fewshot,
                    useGuided: true
                };

                // Start polling for progress updates
                let progressLine = null;
                const progressInterval = setInterval(async () => {
                    try {
                        const statusResponse = await fetch('/parsing_status');
                        const status = await statusResponse.json();

                        if (status.active) {
                            const progressText = `Parsing ${status.current}/${status.total}: ${status.current_line}`;

                            if (!progressLine) {
                                // Add new progress line
                                logContent.innerHTML += `<p id="parsing-progress-line" style="color: #b45309;">${progressText}</p>`;
                                progressLine = document.getElementById('parsing-progress-line');
                            } else {
                                // Update existing progress line
                                progressLine.textContent = progressText;
                            }

                            logContent.scrollTop = logContent.scrollHeight;
                        }
                    } catch (e) {
                        // Ignore errors during status polling
                    }
                }, 500);

                // Call the parsing endpoint
                logContent.innerHTML += '<p style="color: var(--teal);"><i class="bi bi-check-circle mr-1"></i> Model loaded, starting parsing...</p>';
                logContent.scrollTop = logContent.scrollHeight;

                const response = await fetch('/api/parse_structured', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                // Stop progress polling
                clearInterval(progressInterval);
                if (progressLine) {
                    progressLine.remove();
                }

                if (!response.ok) {
                    throw new Error(`Server error: ${response.statusText}`);
                }

                // Download the Excel file
                logContent.innerHTML += '<p style="color: var(--teal);"><i class="bi bi-check-circle-fill mr-1"></i> Parsing complete! Preparing Excel download...</p>';
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                const timestamp = new Date().toISOString().split('T')[0];
                a.download = `parsed_output_${timestamp}.xlsx`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                logContent.innerHTML += '<p style="color: #4ade80; font-weight: bold;"><i class="bi bi-file-earmark-excel-fill mr-1"></i> Excel file downloaded successfully!</p>';
                logContent.scrollTop = logContent.scrollHeight;

                alert('Parsing complete! Excel file downloaded successfully.');

            } catch (error) {
                console.error('Error during parsing:', error);
                logContent.innerHTML += `<p style="color: #b91c1c;"><i class="bi bi-x-circle-fill mr-1"></i> Error: ${error.message}</p>`;
                logContent.innerHTML += '<p style="color: #b45309;"><i class="bi bi-exclamation-triangle mr-1"></i> Make sure the OCR server is running</p>';
                alert(`Error during parsing: ${error.message}\n\nMake sure the OCR server is running.`);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-play-circle-fill mr-1"></i> Run Parsing';
            }
        });

        // Initialize parser tab
        renderTags();
        renderPopupTags();

        // Style active/inactive tabs aligned with PRODUCT.md Terracotta palette
        const style = document.createElement('style');
        style.textContent = `
            .tab-btn.active { background-color: var(--primary-soft); color: var(--primary); border-bottom-color: var(--primary); }
            .tab-btn:not(.active) { background-color: transparent; color: var(--text-dim); }
            .tab-btn:disabled { opacity: 0.45; cursor: not-allowed; }
            .tab-btn:not(:disabled):hover { background-color: rgba(194, 65, 12, 0.04); color: var(--primary); }
        `;
        document.head.appendChild(style);
