let puzzleStartTime = null;

document.addEventListener('DOMContentLoaded', () => {
    const submitBtn = document.getElementById('submit-answer');
    const userAnswerInput = document.getElementById('user-answer');
    const puzzleImage = document.getElementById('puzzle-image');
    const puzzleImageContainer = document.querySelector('.puzzle-image-container');
    const resultMessage = document.getElementById('result-message');
    const totalCount = document.getElementById('total-count');
    const correctCount = document.getElementById('correct-count');
    const accuracyEl = document.getElementById('accuracy');
    const puzzlePrompt = document.getElementById('puzzle-prompt');
    const inputGroup = document.querySelector('.input-group');

    const benchmarkStats = { total: 0, correct: 0 };

    let currentPuzzle = null;
    let bingoSelectedCells = [];
    let shadowSelectedCells = [];
    let mirrorSelectedCells = [];
    let squiggleSelectedIndex = null;
    let transformPipelineSelectedIndex = null;
    let spookyGridSelectedCells = [];
    let storyboardOrder = [];
    let storyboardSelectedIndices = [];
    let jigsawPlacements = [];
    let squiggleRevealTimeout = null;
    let colorCipherRevealTimeout = null;
    let redDotTimeout = null;
    let redDotAnswered = false;
    let redDotHits = 0;
    let redDotRequiredHits = 0;
    let redDotTimeoutDuration = 2000;
    let redDotElement = null;
    let spookySizeAnswered = false;

    submitBtn.addEventListener('click', submitAnswer);
    userAnswerInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            submitAnswer();
        }
    });

    displayDifficultyStars('Dice_Count');
    loadNewPuzzle();

    function resetInterface() {
        bingoSelectedCells = [];
        shadowSelectedCells = [];
        mirrorSelectedCells = [];
        squiggleSelectedIndex = null;
        transformPipelineSelectedIndex = null;
        storyboardOrder = [];
        storyboardSelectedIndices = [];
        jigsawPlacements = [];
        if (squiggleRevealTimeout) {
            clearTimeout(squiggleRevealTimeout);
            squiggleRevealTimeout = null;
        }
        if (colorCipherRevealTimeout) {
            clearTimeout(colorCipherRevealTimeout);
            colorCipherRevealTimeout = null;
        }
        if (redDotTimeout) {
            clearTimeout(redDotTimeout);
            redDotTimeout = null;
        }
        redDotAnswered = false;
        redDotHits = 0;
        redDotRequiredHits = 0;
        redDotTimeoutDuration = 2000;
        redDotElement = null;
        spookySizeAnswered = false;

        if (inputGroup) {
            inputGroup.style.display = 'flex';
        }

        userAnswerInput.type = 'text';
        userAnswerInput.value = '';
        userAnswerInput.placeholder = 'Your answer';
        userAnswerInput.style.display = 'block';

        submitBtn.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit';

        resultMessage.textContent = '';
        resultMessage.className = 'result-message';

        puzzleImageContainer.innerHTML = '';
        puzzleImageContainer.style.display = '';
        puzzleImageContainer.style.width = '';
        puzzleImageContainer.style.maxWidth = '';
        puzzleImageContainer.style.margin = '';
        if (puzzleImageContainer) {
            puzzleImageContainer.classList.remove('adversarial-layout');
        }

        puzzleImage.style.display = 'none';
        puzzleImage.src = '';

        const customSelectors = [
            '.bingo-grid',
            '.bingo-submit',
            '.shadow-plausible-grid',
            '.shadow-submit',
            '.mirror-layout',
            '.mirror-submit',
            '.squiggle-preview',
            '.squiggle-options-grid',
            '.squiggle-submit',
            '.transform-pipeline-container',
            '.transform-pipeline-submit',
            '.color-cipher-preview',
            '.color-cipher-question',
            '.red-dot-area',
            '.trajectory-gif-container',
            '.storyboard-logic-container',
            '.jigsaw-puzzle-container'
        ];

        customSelectors.forEach((selector) => {
            document.querySelectorAll(selector).forEach((element) => element.remove());
        });
    }

    function submitRedDotAttempt(redDotAnswer) {
        if (!currentPuzzle || currentPuzzle.input_type !== 'red_dot_click') {
            return;
        }

        const answerData = {
            puzzle_type: currentPuzzle.puzzle_type,
            puzzle_id: currentPuzzle.puzzle_id,
            answer: redDotAnswer
        };
        answerData.elapsed_time = ((Date.now() - (puzzleStartTime || Date.now())) / 1000).toFixed(2);

        fetch('/api/check_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(answerData)
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.status === 'continue') {
                    redDotHits = Number.isFinite(data.hits_completed) ? data.hits_completed : redDotHits;
                    redDotRequiredHits = Number.isFinite(data.required_hits) ? data.required_hits : redDotRequiredHits;
                    redDotTimeoutDuration = Number.isFinite(data.timeout_ms) ? data.timeout_ms : redDotTimeoutDuration;

                    const nextDot = data.next_dot || {};
                    currentPuzzle.dot = nextDot;
                    currentPuzzle.timeout_ms = redDotTimeoutDuration;
                    currentPuzzle.required_hits = redDotRequiredHits;
                    currentPuzzle.hits_completed = redDotHits;
                    if (redDotElement) {
                        if (Number.isFinite(nextDot.diameter)) {
                            redDotElement.style.width = `${nextDot.diameter}px`;
                            redDotElement.style.height = `${nextDot.diameter}px`;
                        }
                        if (Number.isFinite(nextDot.x)) {
                            redDotElement.style.left = `${nextDot.x}px`;
                        }
                        if (Number.isFinite(nextDot.y)) {
                            redDotElement.style.top = `${nextDot.y}px`;
                        }
                        redDotElement.classList.remove('red-dot-hidden');
                    }

                    redDotAnswered = false;
                    displayRedDotProgress();
                    scheduleRedDotTimeout(redDotTimeoutDuration);
                    return;
                }

                benchmarkStats.total += 1;

                if (data.correct) {
                    benchmarkStats.correct += 1;
                    redDotHits = redDotRequiredHits;
                    resultMessage.textContent = 'Correct!';
                    resultMessage.className = 'result-message correct';
                    createFireworks();
                } else {
                    const failureMessage = data.message || 'Incorrect.';
                    resultMessage.textContent = failureMessage;
                    resultMessage.className = 'result-message incorrect';
                    createSadFace();
                }

                updateStats();
                recordBenchmarkResult({
                    puzzle_type: currentPuzzle.puzzle_type,
                    puzzle_id: currentPuzzle.puzzle_id,
                    user_answer: redDotAnswer,
                    correct_answer: data.correct_answer,
                    correct: data.correct,
                    elapsed_time: answerData.elapsed_time
                });

                setTimeout(loadNewPuzzle, 2000);
            })
            .catch((error) => {
                console.error('Error checking red dot answer:', error);
                showError('Error checking answer. Please try again.');
            });
    }

    function renderPuzzleMedia(data) {
        const mediaPath = data.media_path || data.image_path;
        if (!mediaPath) {
            return;
        }

        const mediaType = (data.media_type || 'image').toLowerCase();
        if (mediaType === 'video') {
            const video = document.createElement('video');
            video.className = 'puzzle-video';
            video.src = mediaPath;
            video.autoplay = true;
            video.loop = true;
            video.muted = true;
            video.playsInline = true;
            video.controls = true;
            video.setAttribute('preload', 'auto');
            puzzleImageContainer.appendChild(video);
        } else {
            puzzleImage.src = mediaPath;
            puzzleImage.alt = data.media_alt || data.prompt || 'CAPTCHA Puzzle';
            puzzleImage.style.display = 'block';
            puzzleImageContainer.appendChild(puzzleImage);
        }
    }

    function loadNewPuzzle() {
        resetInterface();
        puzzlePrompt.textContent = 'Loading puzzle...';

        fetch('/api/get_puzzle?mode=sequential')
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    throw new Error(data.error);
                }

                currentPuzzle = data;
                puzzleStartTime = Date.now();

                displayDifficultyStars(data.puzzle_type);
                puzzlePrompt.textContent = data.prompt || 'Solve the CAPTCHA puzzle';
                if (puzzleImageContainer) {
                    const isAdversarial = data.puzzle_type === 'Adversarial';
                    puzzleImageContainer.classList.toggle('adversarial-layout', isAdversarial);
                }

                switch (data.input_type) {
                    case 'number':
                        configureNumberPuzzle(data);
                        break;
                    case 'bingo_swap':
                        setupBingoSwap(data);
                        break;
                    case 'shadow_plausible':
                        setupShadowPlausibleGrid(data);
                break;
            case 'mirror_select':
                setupMirrorSelect(data);
                break;
            case 'squiggle_select':
                setupSquiggleSelect(data);
                break;
            case 'color_cipher':
                setupColorCipher(data);
                break;
            case 'red_dot_click':
                setupRedDotClick(data);
                break;
            case 'spooky_size_click':
                setupSpookySizeClick(data);
                break;
            case 'storyboard_logic':
                setupStoryboardLogic(data);
                break;
            case 'jigsaw_puzzle':
                setupJigsawPuzzle(data);
                break;
            case 'transform_pipeline_select':
                setupTransformPipelineSelect(data);
                break;
            case 'circle_grid_select':
            case 'circle_grid_direction_select':
            case 'shape_grid_select':
            case 'color_counting_select':
            case 'trajectory_recovery_select':
                setupSpookyGridSelect(data);
                break;
            default:
                configureTextPuzzle(data);
                break;
        }
            })
            .catch((error) => {
                console.error('Error loading puzzle:', error);
                showError('Unable to load a new puzzle. Please refresh the page.');
            });
    }

    function setupRedDotClick(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        redDotAnswered = false;
        redDotHits = Number.isFinite(data?.hits_completed) ? data.hits_completed : 0;
        redDotRequiredHits = Number.isFinite(data?.required_hits) ? data.required_hits : 1;

        const areaWidth = Number.isFinite(data?.area?.width) ? data.area.width : 420;
        const areaHeight = Number.isFinite(data?.area?.height) ? data.area.height : 320;
        const dotDiameter = Number.isFinite(data?.dot?.diameter) ? data.dot.diameter : 48;
        const dotX = Number.isFinite(data?.dot?.x) ? data.dot.x : (areaWidth - dotDiameter) / 2;
        const dotY = Number.isFinite(data?.dot?.y) ? data.dot.y : (areaHeight - dotDiameter) / 2;
        const timeoutMs = Number.isFinite(data?.timeout_ms) ? data.timeout_ms : 2000;
        redDotTimeoutDuration = timeoutMs;

        const area = document.createElement('div');
        area.className = 'red-dot-area';
        area.style.width = `${areaWidth}px`;
        area.style.height = `${areaHeight}px`;

        const dot = document.createElement('div');
        dot.className = 'red-dot';
        dot.style.width = `${dotDiameter}px`;
        dot.style.height = `${dotDiameter}px`;
        dot.style.left = `${dotX}px`;
        dot.style.top = `${dotY}px`;

        area.appendChild(dot);
        puzzleImageContainer.appendChild(area);

        redDotElement = dot;

        const handleSuccessClick = (event) => {
            if (redDotAnswered) {
                return;
            }
            event.stopPropagation();
            const areaRect = area.getBoundingClientRect();
            const clickX = event.clientX - areaRect.left;
            const clickY = event.clientY - areaRect.top;

            finalizeRedDotAttempt({
                clicked: true,
                position: {
                    x: Number.isFinite(clickX) ? Number(clickX.toFixed(2)) : clickX,
                    y: Number.isFinite(clickY) ? Number(clickY.toFixed(2)) : clickY
                },
                relative_position: {
                    x: Number((clickX / areaWidth).toFixed(4)),
                    y: Number((clickY / areaHeight).toFixed(4))
                }
            });
        };

        dot.addEventListener('click', handleSuccessClick);

        scheduleRedDotTimeout(timeoutMs);
        displayRedDotProgress();
    }

    function scheduleRedDotTimeout(duration) {
        if (redDotTimeout) {
            clearTimeout(redDotTimeout);
        }
        redDotTimeout = window.setTimeout(() => {
            if (redDotAnswered) {
                return;
            }
            if (redDotElement) {
                redDotElement.classList.add('red-dot-hidden');
            }
            finalizeRedDotAttempt({ clicked: false });
        }, duration);
    }

    function displayRedDotProgress() {
        if (redDotRequiredHits <= 1) {
            resultMessage.textContent = 'Click the red dot before it disappears!';
        } else {
            resultMessage.textContent = `Click the red dot before it disappears! (${redDotHits}/${redDotRequiredHits})`;
        }
        resultMessage.className = 'result-message instruction';
    }

    function setupSpookySizeClick(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        const canvasWidth = data.canvas_width || 600;
        const canvasHeight = data.canvas_height || 400;

        // Create clickable canvas overlay for the GIF
        const clickArea = document.createElement('div');
        clickArea.className = 'spooky-size-click-area';
        clickArea.style.width = `${canvasWidth}px`;
        clickArea.style.height = `${canvasHeight}px`;
        clickArea.style.position = 'relative';
        clickArea.style.margin = '0 auto';
        clickArea.style.cursor = 'crosshair';
        clickArea.style.border = '2px solid #333';
        clickArea.style.backgroundColor = '#000';

        // Add the GIF as background or img element
        const gifImg = document.createElement('img');
        gifImg.src = data.media_path;
        gifImg.alt = 'Spooky Size Puzzle';
        gifImg.style.width = '100%';
        gifImg.style.height = '100%';
        gifImg.style.display = 'block';
        gifImg.style.pointerEvents = 'none'; // Let clicks pass through to parent

        clickArea.appendChild(gifImg);

        // Handle click
        clickArea.addEventListener('click', (event) => {
            if (spookySizeAnswered) {
                return;
            }

            const rect = clickArea.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;

            // Visual feedback
            const marker = document.createElement('div');
            marker.style.position = 'absolute';
            marker.style.left = `${clickX}px`;
            marker.style.top = `${clickY}px`;
            marker.style.width = '20px';
            marker.style.height = '20px';
            marker.style.marginLeft = '-10px';
            marker.style.marginTop = '-10px';
            marker.style.borderRadius = '50%';
            marker.style.border = '3px solid #0078ff';
            marker.style.backgroundColor = 'rgba(0, 120, 255, 0.3)';
            marker.style.pointerEvents = 'none';
            clickArea.appendChild(marker);

            // Disable further clicks
            clickArea.style.pointerEvents = 'none';
            spookySizeAnswered = true;

            // Submit answer
            const answerData = {
                puzzle_type: currentPuzzle.puzzle_type,
                puzzle_id: currentPuzzle.puzzle_id,
                answer: {
                    position: {
                        x: Number(clickX.toFixed(2)),
                        y: Number(clickY.toFixed(2))
                    }
                }
            };
            answerData.elapsed_time = ((Date.now() - (puzzleStartTime || Date.now())) / 1000).toFixed(2);

            fetch('/api/check_answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(answerData)
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || `HTTP error! status: ${response.status}`);
                    });
                }
                return response.json();
            })
            .then(result => {
                if (result.error) {
                    throw new Error(result.error);
                }

                benchmarkStats.total += 1;

                if (result.correct) {
                    benchmarkStats.correct += 1;
                    resultMessage.textContent = 'Correct! You clicked the right shape.';
                    resultMessage.className = 'result-message correct';
                    createFireworks();
                } else {
                    resultMessage.textContent = 'Incorrect. Try the next puzzle.';
                    resultMessage.className = 'result-message incorrect';
                    createSadFace();
                }

                updateStats();
                recordBenchmarkResult({
                    puzzle_type: currentPuzzle.puzzle_type,
                    puzzle_id: currentPuzzle.puzzle_id,
                    user_answer: answerData.answer,
                    correct_answer: result.correct_answer,
                    correct: result.correct,
                    elapsed_time: answerData.elapsed_time
                });

                setTimeout(() => loadNewPuzzle(), 2000);
            })
            .catch(error => {
                console.error('Error submitting answer:', error);
                resultMessage.textContent = `Error: ${error.message || 'Error submitting answer.'}`;
                resultMessage.className = 'result-message incorrect';
            });
        });

        puzzleImageContainer.appendChild(clickArea);
        puzzleImageContainer.style.display = 'block';
    }

    function setupStoryboardLogic(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        // Initialize order: start with shuffled order to make it interesting
        const images = data.images || [];
        if (!images.length) {
            showError('No storyboard images available.');
            return;
        }

        // Start with images in random order (for challenge)
        storyboardOrder = Array.from({ length: images.length }, (_, i) => i);
        // Shuffle the order
        for (let i = storyboardOrder.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [storyboardOrder[i], storyboardOrder[j]] = [storyboardOrder[j], storyboardOrder[i]];
        }

        // Reset selection
        storyboardSelectedIndices = [];

        const container = document.createElement('div');
        container.className = 'storyboard-logic-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.gap = '20px';
        container.style.margin = '20px auto';
        container.style.maxWidth = '900px';

        const instruction = document.createElement('div');
        instruction.style.fontSize = '16px';
        instruction.style.fontWeight = '500';
        instruction.style.marginBottom = '10px';
        instruction.style.textAlign = 'center';
        instruction.textContent = 'Click two images to swap their positions. Arrange them in the correct story sequence.';
        container.appendChild(instruction);

        const imageRow = document.createElement('div');
        imageRow.style.display = 'flex';
        imageRow.style.gap = '15px';
        imageRow.style.justifyContent = 'center';
        imageRow.style.flexWrap = 'nowrap';
        imageRow.style.width = '100%';
        imageRow.style.alignItems = 'flex-start';

        const renderImages = () => {
            imageRow.innerHTML = '';
            storyboardOrder.forEach((imageIndex, position) => {
                const imageWrapper = document.createElement('div');
                imageWrapper.style.position = 'relative';
                imageWrapper.style.display = 'flex';
                imageWrapper.style.flexDirection = 'column';
                imageWrapper.style.alignItems = 'center';
                imageWrapper.style.cursor = 'pointer';
                imageWrapper.style.transition = 'transform 0.2s';
                imageWrapper.dataset.index = imageIndex;
                imageWrapper.dataset.position = position;

                const positionLabel = document.createElement('div');
                positionLabel.style.position = 'absolute';
                positionLabel.style.top = '-25px';
                positionLabel.style.fontSize = '14px';
                positionLabel.style.fontWeight = '600';
                positionLabel.style.color = '#0078ff';
                positionLabel.textContent = `${position + 1}`;
                imageWrapper.appendChild(positionLabel);

                const img = document.createElement('img');
                img.src = images[imageIndex];
                img.alt = `Storyboard image ${imageIndex + 1}`;
                img.style.width = '250px';
                img.style.height = 'auto';
                img.style.maxWidth = '250px';
                img.style.minWidth = '200px';
                img.style.flexShrink = '0';
                img.style.border = 'none';
                img.style.borderRadius = '8px';
                img.draggable = false;
                imageWrapper.appendChild(img);

                // Set initial border styling
                imageWrapper.style.border = '3px solid #333';
                imageWrapper.style.borderRadius = '8px';
                imageWrapper.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
                imageWrapper.style.padding = '0';
                
                // Check if this position is selected
                const isSelected = storyboardSelectedIndices.includes(position);
                if (isSelected) {
                    imageWrapper.style.border = '3px solid #0078ff';
                    imageWrapper.style.boxShadow = '0 0 0 3px rgba(0, 120, 255, 0.3), 0 2px 8px rgba(0,0,0,0.1)';
                }

                imageWrapper.addEventListener('click', () => {
                    const clickedPosition = position;
                    
                    // If already selected, deselect it
                    if (storyboardSelectedIndices.includes(clickedPosition)) {
                        storyboardSelectedIndices = storyboardSelectedIndices.filter(idx => idx !== clickedPosition);
                        renderImages();
                        return;
                    }
                    
                    // If no selection yet, select this position
                    if (storyboardSelectedIndices.length === 0) {
                        storyboardSelectedIndices.push(clickedPosition);
                        renderImages();
                        return;
                    }
                    
                    // If one position is already selected, swap with this one
                    if (storyboardSelectedIndices.length === 1) {
                        const firstPos = storyboardSelectedIndices[0];
                        const secondPos = clickedPosition;
                        
                        // Swap the images at these positions
                        const temp = storyboardOrder[firstPos];
                        storyboardOrder[firstPos] = storyboardOrder[secondPos];
                        storyboardOrder[secondPos] = temp;
                        
                        // Clear selection
                        storyboardSelectedIndices = [];
                        renderImages();
                    }
                });

                imageWrapper.addEventListener('mouseenter', () => {
                    if (!storyboardSelectedIndices.includes(position)) {
                        imageWrapper.style.transform = 'scale(1.05)';
                    }
                });

                imageWrapper.addEventListener('mouseleave', () => {
                    imageWrapper.style.transform = 'scale(1)';
                });

                imageRow.appendChild(imageWrapper);
            });
        };

        renderImages();
        container.appendChild(imageRow);

        const submitSection = document.createElement('div');
        submitSection.style.marginTop = '20px';

        const storyboardSubmitBtn = document.createElement('button');
        storyboardSubmitBtn.textContent = 'Submit Order';
        storyboardSubmitBtn.className = 'submit-storyboard';
        storyboardSubmitBtn.style.padding = '12px 24px';
        storyboardSubmitBtn.style.fontSize = '16px';
        storyboardSubmitBtn.style.fontWeight = '600';
        storyboardSubmitBtn.style.backgroundColor = '#0078ff';
        storyboardSubmitBtn.style.color = 'white';
        storyboardSubmitBtn.style.border = 'none';
        storyboardSubmitBtn.style.borderRadius = '6px';
        storyboardSubmitBtn.style.cursor = 'pointer';
        storyboardSubmitBtn.style.transition = 'background-color 0.2s';
        storyboardSubmitBtn.type = 'button';

        storyboardSubmitBtn.addEventListener('mouseenter', () => {
            storyboardSubmitBtn.style.backgroundColor = '#0056b3';
        });

        storyboardSubmitBtn.addEventListener('mouseleave', () => {
            storyboardSubmitBtn.style.backgroundColor = '#0078ff';
        });

        storyboardSubmitBtn.addEventListener('click', () => {
            storyboardSubmitBtn.disabled = true;
            storyboardSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(storyboardSubmitBtn);
        container.appendChild(submitSection);

        puzzleImageContainer.appendChild(container);
        puzzleImageContainer.style.display = 'block';
    }

    function setupJigsawPuzzle(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        const pieces = data.pieces || [];
        const gridSize = data.grid_size || [2, 2];
        const pieceSize = data.piece_size || 150;
        const correctPositions = data.correct_positions || [];
        const referenceImage = data.reference_image;

        if (!pieces.length) {
            showError('No puzzle pieces available.');
            return;
        }

        // Initialize placements - all pieces start unplaced
        jigsawPlacements = [];

        const container = document.createElement('div');
        container.className = 'jigsaw-puzzle-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.gap = '20px';
        container.style.margin = '20px auto';
        container.style.maxWidth = '900px';

        // Reference image (optional hint)
        if (referenceImage) {
            const referenceSection = document.createElement('div');
            referenceSection.style.textAlign = 'center';
            referenceSection.style.marginBottom = '10px';
            
            const referenceLabel = document.createElement('div');
            referenceLabel.style.fontSize = '14px';
            referenceLabel.style.fontWeight = '500';
            referenceLabel.style.marginBottom = '5px';
            referenceLabel.textContent = 'Reference image:';
            referenceSection.appendChild(referenceLabel);

            const refImg = document.createElement('img');
            refImg.src = referenceImage;
            refImg.alt = 'Jigsaw puzzle reference';
            refImg.style.maxWidth = `${pieceSize * gridSize[1]}px`;
            refImg.style.height = 'auto';
            refImg.style.border = '2px solid #333';
            refImg.style.borderRadius = '8px';
            refImg.style.opacity = '0.7';
            refImg.draggable = false;
            referenceSection.appendChild(refImg);
            container.appendChild(referenceSection);
        }

        // Puzzle grid area
        const gridContainer = document.createElement('div');
        gridContainer.style.display = 'grid';
        gridContainer.style.gridTemplateColumns = `repeat(${gridSize[1]}, ${pieceSize}px)`;
        gridContainer.style.gridTemplateRows = `repeat(${gridSize[0]}, ${pieceSize}px)`;
        gridContainer.style.gap = '2px';
        gridContainer.style.border = '3px solid #333';
        gridContainer.style.padding = '5px';
        gridContainer.style.backgroundColor = '#f0f0f0';
        gridContainer.style.borderRadius = '8px';
        gridContainer.id = 'jigsaw-grid';

        // Create grid cells
        const gridCells = [];
        for (let row = 0; row < gridSize[0]; row++) {
            for (let col = 0; col < gridSize[1]; col++) {
                const cell = document.createElement('div');
                cell.className = 'jigsaw-grid-cell';
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.style.width = `${pieceSize}px`;
                cell.style.height = `${pieceSize}px`;
                cell.style.border = '2px dashed #ccc';
                cell.style.borderRadius = '4px';
                cell.style.backgroundColor = '#fff';
                cell.style.display = 'flex';
                cell.style.alignItems = 'center';
                cell.style.justifyContent = 'center';
                cell.style.position = 'relative';
                cell.style.transition = 'background-color 0.2s';

                // Drop zone
                cell.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    if (!cell.querySelector('.jigsaw-piece')) {
                        cell.style.backgroundColor = '#e8f4f8';
                    }
                });

                cell.addEventListener('dragleave', () => {
                    if (!cell.querySelector('.jigsaw-piece')) {
                        cell.style.backgroundColor = '#fff';
                    }
                });

                cell.addEventListener('drop', (e) => {
                    e.preventDefault();
                    cell.style.backgroundColor = '#fff';
                    
                    const pieceIndex = parseInt(e.dataTransfer.getData('text/plain'));
                    const row = parseInt(cell.dataset.row);
                    const col = parseInt(cell.dataset.col);
                    
                    // If cell already has a piece, don't replace it
                    if (cell.querySelector('.jigsaw-piece')) {
                        return;
                    }
                    
                    // Find existing placement for this piece
                    const existingPlacementIdx = jigsawPlacements.findIndex(p => p.piece_index === pieceIndex);
                    
                    // If piece was already placed in a different cell, clear that cell
                    if (existingPlacementIdx !== -1) {
                        const oldPlacement = jigsawPlacements[existingPlacementIdx];
                        const oldRow = parseInt(oldPlacement.grid_row);
                        const oldCol = parseInt(oldPlacement.grid_col);
                        // Only clear if it's a different cell
                        if (oldRow !== row || oldCol !== col) {
                            const oldCell = document.querySelector(`.jigsaw-grid-cell[data-row="${oldRow}"][data-col="${oldCol}"]`);
                            if (oldCell && oldCell !== cell) {
                                oldCell.innerHTML = '';
                            }
                            // Update the placement to new position
                            jigsawPlacements[existingPlacementIdx] = {
                                piece_index: pieceIndex,
                                grid_row: row,
                                grid_col: col
                            };
                        } else {
                            // Same cell, no change needed
                            return;
                        }
                    } else {
                        // New placement - add to array
                        jigsawPlacements.push({
                            piece_index: pieceIndex,
                            grid_row: row,
                            grid_col: col
                        });
                    }
                    
                    // Remove piece from tray if it was there
                    const trayPiece = document.querySelector(`.jigsaw-tray-piece[data-piece-index="${pieceIndex}"]`);
                    if (trayPiece) {
                        trayPiece.remove();
                    }
                    
                    // Place piece in this cell
                    const pieceImg = document.createElement('img');
                    pieceImg.src = pieces[pieceIndex];
                    pieceImg.className = 'jigsaw-piece';
                    pieceImg.style.width = '100%';
                    pieceImg.style.height = '100%';
                    pieceImg.style.objectFit = 'contain';
                    pieceImg.draggable = true;
                    pieceImg.dataset.pieceIndex = pieceIndex;
                    
                    // Clear cell and add piece
                    cell.innerHTML = '';
                    cell.appendChild(pieceImg);
                    
                    // Make piece draggable again
                    pieceImg.addEventListener('dragstart', (e) => {
                        e.dataTransfer.setData('text/plain', pieceIndex.toString());
                        e.dataTransfer.effectAllowed = 'move';
                    });
                    
                    // Allow removing piece by dragging to tray
                    pieceImg.addEventListener('dragend', (e) => {
                        // Check if dropped outside grid
                        setTimeout(() => {
                            const dropTarget = document.elementFromPoint(e.clientX, e.clientY);
                            if (!dropTarget?.closest('.jigsaw-grid-cell')) {
                                // Return to tray - remove from cell
                                cell.innerHTML = '';
                                const placementIdx = jigsawPlacements.findIndex(p => p.piece_index === pieceIndex);
                                if (placementIdx !== -1) {
                                    jigsawPlacements.splice(placementIdx, 1);
                                }
                                renderPieces();
                            }
                        }, 100);
                    });
                });

                gridContainer.appendChild(cell);
                gridCells.push(cell);
            }
        }

        container.appendChild(gridContainer);

        // Pieces tray
        const trayContainer = document.createElement('div');
        trayContainer.className = 'jigsaw-tray';
        trayContainer.style.display = 'flex';
        trayContainer.style.flexWrap = 'wrap';
        trayContainer.style.gap = '10px';
        trayContainer.style.justifyContent = 'center';
        trayContainer.style.marginTop = '20px';
        trayContainer.style.padding = '15px';
        trayContainer.style.border = '2px dashed #ccc';
        trayContainer.style.borderRadius = '8px';
        trayContainer.style.backgroundColor = '#fafafa';
        trayContainer.style.minHeight = '100px';

        const trayLabel = document.createElement('div');
        trayLabel.style.width = '100%';
        trayLabel.style.textAlign = 'center';
        trayLabel.style.fontSize = '14px';
        trayLabel.style.fontWeight = '500';
        trayLabel.style.marginBottom = '10px';
        trayLabel.textContent = 'Drag pieces from here to the grid above';
        trayContainer.appendChild(trayLabel);

        const renderPieces = () => {
            // Clear tray
            const existingPieces = trayContainer.querySelectorAll('.jigsaw-tray-piece');
            existingPieces.forEach(p => p.remove());

            // Show pieces that are not placed
            const placedPieceIndices = new Set(jigsawPlacements.map(p => p.piece_index));
            
            pieces.forEach((pieceSrc, index) => {
                if (!placedPieceIndices.has(index)) {
                    const pieceWrapper = document.createElement('div');
                    pieceWrapper.className = 'jigsaw-tray-piece';
                    pieceWrapper.dataset.pieceIndex = index;
                    pieceWrapper.style.width = `${pieceSize * 0.6}px`;
                    pieceWrapper.style.height = `${pieceSize * 0.6}px`;
                    pieceWrapper.style.cursor = 'grab';
                    pieceWrapper.style.border = '2px solid #333';
                    pieceWrapper.style.borderRadius = '4px';
                    pieceWrapper.style.overflow = 'hidden';
                    pieceWrapper.style.transition = 'transform 0.2s';
                    pieceWrapper.style.backgroundColor = '#fff';

                    const pieceImg = document.createElement('img');
                    pieceImg.src = pieceSrc;
                    pieceImg.style.width = '100%';
                    pieceImg.style.height = '100%';
                    pieceImg.style.objectFit = 'contain';
                    pieceImg.draggable = true;
                    pieceImg.dataset.pieceIndex = index;

                    pieceWrapper.appendChild(pieceImg);
                    trayContainer.appendChild(pieceWrapper);

                    pieceImg.addEventListener('dragstart', (e) => {
                        e.dataTransfer.setData('text/plain', index.toString());
                        e.dataTransfer.effectAllowed = 'move';
                        pieceWrapper.style.opacity = '0.5';
                    });

                    pieceImg.addEventListener('dragend', () => {
                        pieceWrapper.style.opacity = '1';
                    });

                    pieceWrapper.addEventListener('mouseenter', () => {
                        pieceWrapper.style.transform = 'scale(1.1)';
                    });

                    pieceWrapper.addEventListener('mouseleave', () => {
                        pieceWrapper.style.transform = 'scale(1)';
                    });
                }
            });
        };

        renderPieces();
        container.appendChild(trayContainer);

        // Submit button
        const submitSection = document.createElement('div');
        submitSection.style.marginTop = '20px';

        const jigsawSubmitBtn = document.createElement('button');
        jigsawSubmitBtn.textContent = 'Submit Puzzle';
        jigsawSubmitBtn.className = 'submit-jigsaw';
        jigsawSubmitBtn.style.padding = '12px 24px';
        jigsawSubmitBtn.style.fontSize = '16px';
        jigsawSubmitBtn.style.fontWeight = '600';
        jigsawSubmitBtn.style.backgroundColor = '#0078ff';
        jigsawSubmitBtn.style.color = 'white';
        jigsawSubmitBtn.style.border = 'none';
        jigsawSubmitBtn.style.borderRadius = '6px';
        jigsawSubmitBtn.style.cursor = 'pointer';
        jigsawSubmitBtn.style.transition = 'background-color 0.2s';
        jigsawSubmitBtn.type = 'button';

        jigsawSubmitBtn.addEventListener('mouseenter', () => {
            jigsawSubmitBtn.style.backgroundColor = '#0056b3';
        });

        jigsawSubmitBtn.addEventListener('mouseleave', () => {
            jigsawSubmitBtn.style.backgroundColor = '#0078ff';
        });

        jigsawSubmitBtn.addEventListener('click', () => {
            // Allow submission even if pieces aren't placed - backend will mark as incorrect
            // Validate that all placements have valid coordinates (if any pieces are placed)
            const invalidPlacements = jigsawPlacements.filter(p => 
                p.piece_index === undefined || 
                p.grid_row === undefined || 
                p.grid_col === undefined ||
                isNaN(p.piece_index) ||
                isNaN(p.grid_row) ||
                isNaN(p.grid_col)
            );
            
            // If there are placements but they're invalid, warn but allow submission
            if (invalidPlacements.length > 0 && jigsawPlacements.length > 0) {
                console.error('Invalid placements detected:', invalidPlacements);
                // Still allow submission - backend will handle validation
            }
            
            jigsawSubmitBtn.disabled = true;
            jigsawSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(jigsawSubmitBtn);
        container.appendChild(submitSection);

        puzzleImageContainer.appendChild(container);
        puzzleImageContainer.style.display = 'block';
    }

    function configureNumberPuzzle(data) {
        if (inputGroup) {
            inputGroup.style.display = 'flex';
        }

        userAnswerInput.type = 'number';
        userAnswerInput.value = '';
        userAnswerInput.placeholder = 'Enter total';

        submitBtn.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit';

        renderPuzzleMedia(data);
    }

    function configureTextPuzzle(data) {
        if (inputGroup) {
            inputGroup.style.display = 'flex';
        }

        userAnswerInput.type = 'text';
        userAnswerInput.value = '';
        userAnswerInput.placeholder = 'Enter answer';

        submitBtn.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit';

        renderPuzzleMedia(data);
    }

    function setupBingoSwap(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        bingoSelectedCells = [];

        const gridSize = data.grid_size || [3, 3];
        const [rows, cols] = gridSize;

        const gridContainer = document.createElement('div');
        gridContainer.className = 'bingo-grid';
        gridContainer.style.display = 'grid';
        gridContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        gridContainer.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
        gridContainer.style.gap = '4px';
        gridContainer.style.width = '100%';
        gridContainer.style.maxWidth = '640px';
        gridContainer.style.margin = '0 auto';

        const fullImg = new Image();
        fullImg.onload = () => {
            const cellWidth = fullImg.width / cols;
            const cellHeight = fullImg.height / rows;
            const totalCells = rows * cols;

            for (let i = 0; i < totalCells; i += 1) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.dataset.index = i;
                cell.style.position = 'relative';
                cell.style.border = '2px solid #333';
                cell.style.borderRadius = '6px';
                cell.style.overflow = 'hidden';
                cell.style.cursor = 'pointer';
                cell.style.transition = 'transform 0.2s ease, border-color 0.2s ease';

                const cellImg = document.createElement('img');
                cellImg.className = 'cell-image';
                cellImg.style.width = '100%';
                cellImg.style.height = '100%';
                cellImg.style.objectFit = 'cover';
                cell.appendChild(cellImg);

                const canvas = document.createElement('canvas');
                canvas.width = cellWidth;
                canvas.height = cellHeight;
                const ctx = canvas.getContext('2d');
                const row = Math.floor(i / cols);
                const col = i % cols;
                ctx.drawImage(
                    fullImg,
                    col * cellWidth,
                    row * cellHeight,
                    cellWidth,
                    cellHeight,
                    0,
                    0,
                    cellWidth,
                    cellHeight
                );
                cellImg.src = canvas.toDataURL();

                const overlay = document.createElement('div');
                overlay.className = 'cell-overlay';
                overlay.style.position = 'absolute';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.backgroundColor = 'rgba(0, 120, 255, 0.5)';
                overlay.style.opacity = '0';
                overlay.style.transition = 'opacity 0.2s ease';
                overlay.style.pointerEvents = 'none';
                cell.appendChild(overlay);

                cell.addEventListener('click', () => toggleBingoCellSelection(i, cell));

                gridContainer.appendChild(cell);
            }

            puzzleImageContainer.appendChild(gridContainer);

            const submitSection = document.createElement('div');
            submitSection.className = 'bingo-submit';
            submitSection.style.textAlign = 'center';
            submitSection.style.marginTop = '18px';

            const bingoSubmitBtn = document.createElement('button');
            bingoSubmitBtn.textContent = 'Swap and Submit';
            bingoSubmitBtn.className = 'submit-bingo';
            bingoSubmitBtn.addEventListener('click', () => {
                if (bingoSelectedCells.length !== 2) {
                    showError('Please select exactly two cells to swap.');
                    return;
                }
                swapBingoCells();
                bingoSubmitBtn.disabled = true;
                bingoSubmitBtn.textContent = 'Processing...';
                submitAnswer();
            });

            submitSection.appendChild(bingoSubmitBtn);
            puzzleImageContainer.appendChild(submitSection);
        };

        fullImg.src = data.image_path;
    }

    function toggleBingoCellSelection(index, cellElement) {
        const overlay = cellElement.querySelector('.cell-overlay');

        const selectedIndex = bingoSelectedCells.indexOf(index);
        if (selectedIndex !== -1) {
            bingoSelectedCells.splice(selectedIndex, 1);
            if (overlay) {
                overlay.style.opacity = '0';
            }
            cellElement.style.transform = 'scale(1)';
            cellElement.style.borderColor = '#333';
        } else {
            if (bingoSelectedCells.length === 2) {
                const firstIndex = bingoSelectedCells.shift();
                const firstCell = document.querySelector(`.grid-cell[data-index="${firstIndex}"]`);
                if (firstCell) {
                    const firstOverlay = firstCell.querySelector('.cell-overlay');
                    if (firstOverlay) {
                        firstOverlay.style.opacity = '0';
                    }
                    firstCell.style.transform = 'scale(1)';
                    firstCell.style.borderColor = '#333';
                }
            }

            bingoSelectedCells.push(index);
            if (overlay) {
                overlay.style.opacity = '1';
            }
            cellElement.style.transform = 'scale(0.96)';
            cellElement.style.borderColor = '#0078ff';
        }
    }

    function swapBingoCells() {
        if (bingoSelectedCells.length !== 2) {
            return;
        }

        const [firstIndex, secondIndex] = bingoSelectedCells;
        const firstCell = document.querySelector(`.grid-cell[data-index="${firstIndex}"]`);
        const secondCell = document.querySelector(`.grid-cell[data-index="${secondIndex}"]`);

        if (!firstCell || !secondCell) {
            return;
        }

        const firstImage = firstCell.querySelector('img');
        const secondImage = secondCell.querySelector('img');

        if (firstImage && secondImage) {
            const tempSrc = firstImage.src;
            firstImage.src = secondImage.src;
            secondImage.src = tempSrc;
        }
    }

    function setupShadowPlausibleGrid(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        shadowSelectedCells = [];

        puzzleImageContainer.style.display = 'block';
        puzzleImageContainer.style.width = '100%';
        puzzleImageContainer.style.maxWidth = '960px';
        puzzleImageContainer.style.margin = '0 auto';

        const gridContainer = document.createElement('div');
        gridContainer.className = 'shadow-plausible-grid';

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No shadow options available.');
            return;
        }

        const gridSize = data.grid_size || [];
        const cols = gridSize[1] || Math.ceil(Math.sqrt(optionImages.length));
        const rows = gridSize[0] || Math.ceil(optionImages.length / cols);
        gridContainer.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
        gridContainer.dataset.rows = rows;
        gridContainer.dataset.cols = cols;

        optionImages.forEach((src, index) => {
            const cell = document.createElement('div');
            cell.className = 'shadow-cell';
            cell.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Shadow option ${index + 1}`;
            img.draggable = false;
            cell.appendChild(img);

            const overlay = document.createElement('div');
            overlay.className = 'shadow-overlay';
            cell.appendChild(overlay);

            const checkmark = document.createElement('div');
            checkmark.className = 'shadow-checkmark';
            checkmark.textContent = '✓';
            cell.appendChild(checkmark);

            cell.addEventListener('click', () => toggleShadowSelection(index, cell));

            gridContainer.appendChild(cell);
        });

        puzzleImageContainer.appendChild(gridContainer);

        const submitSection = document.createElement('div');
        submitSection.className = 'shadow-submit';

        const shadowSubmitBtn = document.createElement('button');
        shadowSubmitBtn.textContent = 'Submit';
        shadowSubmitBtn.className = 'submit-shadow';
        shadowSubmitBtn.type = 'button';
        shadowSubmitBtn.addEventListener('click', () => {
            if (!shadowSelectedCells.length) {
                showError('Select at least one image before submitting.');
                return;
            }
            shadowSubmitBtn.disabled = true;
            shadowSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(shadowSubmitBtn);
        puzzleImageContainer.appendChild(submitSection);
    }

    function toggleShadowSelection(index, cellElement) {
        const overlay = cellElement.querySelector('.shadow-overlay');
        const checkmark = cellElement.querySelector('.shadow-checkmark');

        const alreadySelected = shadowSelectedCells.includes(index);
        if (alreadySelected) {
            shadowSelectedCells = shadowSelectedCells.filter((idx) => idx !== index);
            if (overlay) {
                overlay.style.opacity = '0';
            }
            if (checkmark) {
                checkmark.style.opacity = '0';
            }
            cellElement.style.transform = 'scale(1)';
            cellElement.style.borderColor = '#333';
        } else {
            shadowSelectedCells.push(index);
            if (overlay) {
                overlay.style.opacity = '1';
            }
            if (checkmark) {
                checkmark.style.opacity = '1';
            }
            cellElement.style.transform = 'scale(0.97)';
            cellElement.style.borderColor = '#0078ff';
        }
    }

    function setupMirrorSelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        mirrorSelectedCells = [];

        const layout = document.createElement('div');
        layout.className = 'mirror-layout';

        const referenceSection = document.createElement('div');
        referenceSection.className = 'mirror-reference';

        const referenceLabel = document.createElement('div');
        referenceLabel.className = 'mirror-reference-label';
        referenceLabel.textContent = 'Reference';
        referenceSection.appendChild(referenceLabel);

        const referenceImg = document.createElement('img');
        referenceImg.src = data.reference_image;
        referenceImg.alt = 'Reference object';
        referenceImg.draggable = false;
        referenceSection.appendChild(referenceImg);

        const optionsSection = document.createElement('div');
        optionsSection.className = 'mirror-options';

        const optionsLabel = document.createElement('div');
        optionsLabel.className = 'mirror-options-label';
        optionsLabel.textContent = 'Select all incorrect mirrors';
        optionsSection.appendChild(optionsLabel);

        const optionsGrid = document.createElement('div');
        optionsGrid.className = 'mirror-options-grid';

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No mirror options available.');
            return;
        }

        const gridSize = data.grid_size || [1, optionImages.length];
        const cols = gridSize[1] || optionImages.length || 1;
        optionsGrid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;

        optionImages.forEach((src, index) => {
            const cell = document.createElement('div');
            cell.className = 'mirror-option';
            cell.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Mirror option ${index + 1}`;
            img.draggable = false;
            cell.appendChild(img);

            const overlay = document.createElement('div');
            overlay.className = 'mirror-overlay';
            cell.appendChild(overlay);

            const badge = document.createElement('div');
            badge.className = 'mirror-checkmark';
            badge.textContent = '✕';
            cell.appendChild(badge);

            cell.addEventListener('click', () => toggleMirrorSelection(index, cell));

            optionsGrid.appendChild(cell);
        });

        optionsSection.appendChild(optionsGrid);
        layout.appendChild(referenceSection);
        layout.appendChild(optionsSection);
        
        // Reset container display for mirror layout
        puzzleImageContainer.style.display = 'block';
        puzzleImageContainer.appendChild(layout);

        const submitSection = document.createElement('div');
        submitSection.className = 'mirror-submit';

        const mirrorSubmitBtn = document.createElement('button');
        mirrorSubmitBtn.textContent = 'Submit';
        mirrorSubmitBtn.className = 'submit-mirror';
        mirrorSubmitBtn.type = 'button';
        mirrorSubmitBtn.addEventListener('click', () => {
            if (!mirrorSelectedCells.length) {
                showError('Select at least one mirror before submitting.');
                return;
            }
            mirrorSubmitBtn.disabled = true;
            mirrorSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(mirrorSubmitBtn);
        puzzleImageContainer.appendChild(submitSection);
    }

    function setupSpookyGridSelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        // Make sure result message is visible (it's inside inputGroup)
        if (resultMessage) {
            resultMessage.style.display = 'block';
            resultMessage.style.position = 'relative';
            resultMessage.style.marginTop = '20px';
        }

        spookyGridSelectedCells = [];

        puzzleImageContainer.style.display = 'block';
        puzzleImageContainer.style.width = '100%';
        puzzleImageContainer.style.maxWidth = '960px';
        puzzleImageContainer.style.margin = '0 auto';

        // For Trajectory_Recovery, show the movement GIF above the grid
        if (data.puzzle_type === 'Trajectory_Recovery' && data.movement_gif) {
            const gifContainer = document.createElement('div');
            gifContainer.className = 'trajectory-gif-container';
            gifContainer.style.textAlign = 'center';
            gifContainer.style.marginBottom = '20px';

            const gifImg = document.createElement('img');
            gifImg.src = data.movement_gif;
            gifImg.alt = 'Ball movement trajectory';
            gifImg.style.maxWidth = '400px';
            gifImg.style.width = '100%';
            gifImg.style.border = '2px solid #333';
            gifImg.style.borderRadius = '8px';
            gifImg.draggable = false;

            gifContainer.appendChild(gifImg);
            puzzleImageContainer.appendChild(gifContainer);
        }

        const gridContainer = document.createElement('div');
        gridContainer.className = 'spooky-grid-container';

        // Add special class for Color_Counting to have white background
        if (data.puzzle_type === 'Color_Counting') {
            gridContainer.classList.add('color-counting-grid');
        }

        // Add special class for Trajectory_Recovery
        if (data.puzzle_type === 'Trajectory_Recovery') {
            gridContainer.classList.add('trajectory-recovery-grid');
        }

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No spooky grid options available.');
            return;
        }

        const gridSize = data.grid_size || [3, 3];
        const cols = gridSize[1] || 3;
        const rows = gridSize[0] || 3;
        gridContainer.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
        gridContainer.dataset.rows = rows;
        gridContainer.dataset.cols = cols;

        optionImages.forEach((src, index) => {
            const cell = document.createElement('div');
            cell.className = 'spooky-grid-cell';
            cell.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Grid option ${index + 1}`;
            img.draggable = false;
            cell.appendChild(img);

            const overlay = document.createElement('div');
            overlay.className = 'spooky-grid-overlay';
            cell.appendChild(overlay);

            const checkmark = document.createElement('div');
            checkmark.className = 'spooky-grid-checkmark';
            checkmark.textContent = '✓';
            cell.appendChild(checkmark);

            cell.addEventListener('click', () => toggleSpookyGridSelection(index, cell));

            gridContainer.appendChild(cell);
        });

        puzzleImageContainer.appendChild(gridContainer);

        const submitSection = document.createElement('div');
        submitSection.className = 'spooky-grid-submit';

        const spookySubmitBtn = document.createElement('button');
        spookySubmitBtn.textContent = 'Submit';
        spookySubmitBtn.className = 'submit-spooky-grid';
        spookySubmitBtn.type = 'button';
        spookySubmitBtn.addEventListener('click', () => {
            if (!spookyGridSelectedCells.length) {
                showError('Select at least one cell before submitting.');
                return;
            }
            spookySubmitBtn.disabled = true;
            spookySubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(spookySubmitBtn);
        puzzleImageContainer.appendChild(submitSection);
    }

    function toggleSpookyGridSelection(index, cellElement) {
        const overlay = cellElement.querySelector('.spooky-grid-overlay');
        const checkmark = cellElement.querySelector('.spooky-grid-checkmark');

        const alreadySelected = spookyGridSelectedCells.includes(index);
        if (alreadySelected) {
            spookyGridSelectedCells = spookyGridSelectedCells.filter((idx) => idx !== index);
            if (overlay) {
                overlay.style.opacity = '0';
            }
            if (checkmark) {
                checkmark.style.opacity = '0';
            }
            cellElement.style.transform = 'scale(1)';
            cellElement.style.borderColor = '#333';
        } else {
            spookyGridSelectedCells.push(index);
            if (overlay) {
                overlay.style.opacity = '1';
            }
            if (checkmark) {
                checkmark.style.opacity = '1';
            }
            cellElement.style.transform = 'scale(0.97)';
            cellElement.style.borderColor = '#0078ff';
        }
    }


    function setupSquiggleSelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        squiggleSelectedIndex = null;

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No squiggle options available.');
            return;
        }

        const revealDuration = Number.parseInt(data.reveal_duration, 10);
        const revealSeconds = Number.isFinite(revealDuration) && revealDuration > 0 ? revealDuration : 3;

        const previewWrapper = document.createElement('div');
        previewWrapper.className = 'squiggle-preview';

        const previewHint = document.createElement('div');
        previewHint.className = 'squiggle-hint';
        previewHint.textContent = `Memorize the trace. Choices appear in ${revealSeconds} second${revealSeconds === 1 ? '' : 's'}.`;
        previewWrapper.appendChild(previewHint);

        const previewImage = document.createElement('img');
        previewImage.src = data.reference_image;
        previewImage.alt = 'Trace preview';
        previewImage.draggable = false;
        previewImage.className = 'squiggle-preview-image';
        previewWrapper.appendChild(previewImage);

        puzzleImageContainer.appendChild(previewWrapper);

        const optionsGrid = document.createElement('div');
        optionsGrid.className = 'squiggle-options-grid';
        optionsGrid.style.display = 'none';

        // Reset container display for squiggle layout
        puzzleImageContainer.style.display = 'block';
        
        const gridSize = Array.isArray(data.grid_size) ? data.grid_size : null;
        if (gridSize && gridSize.length > 1 && Number.isFinite(gridSize[1]) && gridSize[1] > 0) {
            optionsGrid.style.gridTemplateColumns = `repeat(${gridSize[1]}, minmax(160px, 1fr))`;
        } else if (optionImages.length === 4) {
            optionsGrid.style.gridTemplateColumns = 'repeat(2, minmax(160px, 1fr))';
        }
        optionsGrid.style.columnGap = '40px';
        optionsGrid.style.rowGap = '32px';
        optionsGrid.style.justifyContent = 'center';

        optionImages.forEach((src, index) => {
            const option = document.createElement('div');
            option.className = 'squiggle-option';
            option.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Squiggle option ${index + 1}`;
            img.draggable = false;
            option.appendChild(img);

            option.addEventListener('click', () => selectSquiggleOption(index, option));

            optionsGrid.appendChild(option);
        });

        puzzleImageContainer.appendChild(optionsGrid);

        const submitSection = document.createElement('div');
        submitSection.className = 'squiggle-submit';
        submitSection.style.display = 'none';

        const squiggleSubmitBtn = document.createElement('button');
        squiggleSubmitBtn.className = 'submit-squiggle';
        squiggleSubmitBtn.type = 'button';
        squiggleSubmitBtn.textContent = 'Submit';
        squiggleSubmitBtn.addEventListener('click', () => {
            if (squiggleSelectedIndex === null) {
                showError('Select the squiggle that matches the preview.');
                return;
            }
            squiggleSubmitBtn.disabled = true;
            squiggleSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(squiggleSubmitBtn);
        puzzleImageContainer.appendChild(submitSection);

        squiggleRevealTimeout = setTimeout(() => {
            previewWrapper.remove();
            optionsGrid.style.display = 'grid';
            submitSection.style.display = 'flex';
        }, revealSeconds * 1000);
    }

    function selectSquiggleOption(index, optionElement) {
        if (squiggleSelectedIndex === index) {
            squiggleSelectedIndex = null;
            optionElement.classList.remove('active');
            return;
        }

        const previouslyActive = document.querySelector('.squiggle-option.active');
        if (previouslyActive) {
            previouslyActive.classList.remove('active');
        }

        squiggleSelectedIndex = index;
        optionElement.classList.add('active');
    }

    function setupTransformPipelineSelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        transformPipelineSelectedIndex = null;

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No transform pipeline options available.');
            return;
        }

        const container = document.createElement('div');
        container.className = 'transform-pipeline-container';

        // Reference image section
        const referenceSection = document.createElement('div');
        referenceSection.className = 'transform-pipeline-reference';

        const referenceLabel = document.createElement('div');
        referenceLabel.className = 'transform-pipeline-label';
        referenceLabel.textContent = 'Starting Image:';
        referenceSection.appendChild(referenceLabel);

        const referenceImage = document.createElement('img');
        referenceImage.src = data.reference_image;
        referenceImage.alt = 'Reference image';
        referenceImage.draggable = false;
        referenceImage.className = 'transform-pipeline-ref-image';
        referenceSection.appendChild(referenceImage);

        // Transform steps section
        const stepsSection = document.createElement('div');
        stepsSection.className = 'transform-pipeline-steps';

        const stepsLabel = document.createElement('div');
        stepsLabel.className = 'transform-pipeline-label';
        stepsLabel.textContent = 'Transform Steps:';
        stepsSection.appendChild(stepsLabel);

        const stepsList = document.createElement('div');
        stepsList.className = 'transform-pipeline-steps-list';
        (data.transform_steps || []).forEach((step, idx) => {
            const stepItem = document.createElement('div');
            stepItem.className = 'transform-pipeline-step';
            stepItem.textContent = `${idx + 1}. ${step}`;
            stepsList.appendChild(stepItem);
        });
        stepsSection.appendChild(stepsList);

        container.appendChild(referenceSection);
        container.appendChild(stepsSection);

        // Options grid
        const optionsGrid = document.createElement('div');
        optionsGrid.className = 'transform-pipeline-options-grid';

        const optionsLabel = document.createElement('div');
        optionsLabel.className = 'transform-pipeline-label';
        optionsLabel.textContent = 'Select the correct result:';
        optionsLabel.style.marginTop = '20px';
        container.appendChild(optionsLabel);

        puzzleImageContainer.style.display = 'block';

        const gridSize = Array.isArray(data.grid_size) ? data.grid_size : null;
        if (gridSize && gridSize.length > 1 && Number.isFinite(gridSize[1]) && gridSize[1] > 0) {
            optionsGrid.style.gridTemplateColumns = `repeat(${gridSize[1]}, minmax(160px, 1fr))`;
        } else if (optionImages.length === 4) {
            optionsGrid.style.gridTemplateColumns = 'repeat(2, minmax(160px, 1fr))';
        } else if (optionImages.length === 6) {
            optionsGrid.style.gridTemplateColumns = 'repeat(3, minmax(160px, 1fr))';
        }
        optionsGrid.style.columnGap = '40px';
        optionsGrid.style.rowGap = '32px';
        optionsGrid.style.justifyContent = 'center';
        optionsGrid.style.marginTop = '20px';

        optionImages.forEach((src, index) => {
            const option = document.createElement('div');
            option.className = 'transform-pipeline-option';
            option.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Option ${index + 1}`;
            img.draggable = false;
            option.appendChild(img);

            option.addEventListener('click', () => selectTransformPipelineOption(index, option));

            optionsGrid.appendChild(option);
        });

        container.appendChild(optionsGrid);

        // Submit button
        const submitSection = document.createElement('div');
        submitSection.className = 'transform-pipeline-submit';
        submitSection.style.display = 'flex';
        submitSection.style.justifyContent = 'center';
        submitSection.style.marginTop = '20px';

        const transformSubmitBtn = document.createElement('button');
        transformSubmitBtn.className = 'submit-transform-pipeline';
        transformSubmitBtn.type = 'button';
        transformSubmitBtn.textContent = 'Submit';
        transformSubmitBtn.addEventListener('click', () => {
            if (transformPipelineSelectedIndex === null) {
                showError('Select the correct transformed image.');
                return;
            }
            transformSubmitBtn.disabled = true;
            transformSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(transformSubmitBtn);
        container.appendChild(submitSection);

        puzzleImageContainer.appendChild(container);
    }

    function selectTransformPipelineOption(index, optionElement) {
        if (transformPipelineSelectedIndex === index) {
            transformPipelineSelectedIndex = null;
            optionElement.classList.remove('active');
            return;
        }

        const previouslyActive = document.querySelector('.transform-pipeline-option.active');
        if (previouslyActive) {
            previouslyActive.classList.remove('active');
        }

        transformPipelineSelectedIndex = index;
        optionElement.classList.add('active');
    }

    function setupColorCipher(data) {
        const revealDuration = Number.parseInt(data.reveal_duration, 10);
        const revealSeconds = Number.isFinite(revealDuration) && revealDuration > 0 ? revealDuration : 3;

        if (inputGroup) {
            inputGroup.style.display = 'none';
        }

        submitBtn.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.textContent = 'Submit';

        userAnswerInput.type = data.input_mode === 'text' ? 'text' : 'number';
        userAnswerInput.value = '';
        userAnswerInput.placeholder = 'Enter answer';

        const previewWrapper = document.createElement('div');
        previewWrapper.className = 'color-cipher-preview';

        const previewTitle = document.createElement('div');
        previewTitle.className = 'color-cipher-title';
        previewTitle.textContent = 'Remember these values:';
        previewWrapper.appendChild(previewTitle);

        const mappingList = document.createElement('div');
        mappingList.className = 'color-cipher-mapping';

        (data.mapping || []).forEach((item) => {
            const row = document.createElement('div');
            row.className = 'color-cipher-row';

            const symbol = document.createElement('span');
            symbol.className = 'color-cipher-symbol';
            symbol.textContent = item.symbol || '';

            const value = document.createElement('span');
            value.className = 'color-cipher-value';
            value.textContent = `= ${item.value}`;

            row.appendChild(symbol);
            row.appendChild(value);
            mappingList.appendChild(row);
        });

        previewWrapper.appendChild(mappingList);
        puzzleImageContainer.appendChild(previewWrapper);

        const questionBlock = document.createElement('div');
        questionBlock.className = 'color-cipher-question';
        questionBlock.textContent = '';
        questionBlock.style.display = 'none';
        puzzleImageContainer.appendChild(questionBlock);

        colorCipherRevealTimeout = setTimeout(() => {
            previewWrapper.remove();
            if (inputGroup) {
                inputGroup.style.display = 'flex';
            }
            submitBtn.disabled = false;
            // questionBlock.textContent = data.question || 'What is the answer?';
            questionBlock.style.display = 'block';
            puzzlePrompt.textContent = data.question || 'What is the answer?';
            userAnswerInput.focus();
        }, revealSeconds * 1000);
    }

    function finalizeRedDotAttempt(answerPayload) {
        if (redDotAnswered) {
            return;
        }
        redDotAnswered = true;
        if (redDotTimeout) {
            clearTimeout(redDotTimeout);
            redDotTimeout = null;
        }

        if (redDotElement) {
            redDotElement.classList.add('red-dot-hidden');
        }

        const payload = {
            ...answerPayload,
            hit_index: redDotHits
        };

        submitRedDotAttempt(payload);
    }

    function toggleMirrorSelection(index, cellElement) {
        const overlay = cellElement.querySelector('.mirror-overlay');
        const badge = cellElement.querySelector('.mirror-checkmark');

        const alreadySelected = mirrorSelectedCells.includes(index);
        if (alreadySelected) {
            mirrorSelectedCells = mirrorSelectedCells.filter((idx) => idx !== index);
            if (overlay) {
                overlay.classList.remove('active');
            }
            if (badge) {
                badge.classList.remove('active');
            }
            cellElement.classList.remove('active');
        } else {
            mirrorSelectedCells.push(index);
            if (overlay) {
                overlay.classList.add('active');
            }
            if (badge) {
                badge.classList.add('active');
            }
            cellElement.classList.add('active');
        }
    }

    function submitAnswer(overrideAnswer = undefined) {
        if (!currentPuzzle) {
            return;
        }

        if (currentPuzzle.input_type === 'red_dot_click') {
            return;
        }

        if ((currentPuzzle.input_type === 'number' || currentPuzzle.input_type === 'text') &&
            !userAnswerInput.value.trim()) {
            return;
        }

        const answerData = {
            puzzle_type: currentPuzzle.puzzle_type,
            puzzle_id: currentPuzzle.puzzle_id
        };

        switch (currentPuzzle.input_type) {
            case 'number':
            case 'text':
                answerData.answer = userAnswerInput.value.trim();
                break;
            case 'bingo_swap':
                answerData.answer = bingoSelectedCells;
                if (bingoSelectedCells.length !== 2) {
                    showError('Please select exactly two cells to swap.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'shadow_plausible':
                answerData.answer = shadowSelectedCells;
                if (!shadowSelectedCells.length) {
                    showError('Select at least one image before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'mirror_select':
                answerData.answer = mirrorSelectedCells;
                if (!mirrorSelectedCells.length) {
                    showError('Select at least one mirror before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'squiggle_select':
                answerData.answer = squiggleSelectedIndex;
                if (squiggleSelectedIndex === null) {
                    showError('Select the squiggle that matches the preview.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'transform_pipeline_select':
                answerData.answer = transformPipelineSelectedIndex;
                if (transformPipelineSelectedIndex === null) {
                    showError('Select the correct transformed image.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'color_cipher':
                if (!userAnswerInput.value.trim()) {
                    showError('Enter your answer before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                answerData.answer = userAnswerInput.value.trim();
                if (currentPuzzle.cipher_state) {
                    answerData.cipher_state = currentPuzzle.cipher_state;
                }
                break;
            case 'circle_grid_select':
            case 'circle_grid_direction_select':
            case 'shape_grid_select':
            case 'color_counting_select':
            case 'trajectory_recovery_select':
                answerData.answer = spookyGridSelectedCells;
                if (!spookyGridSelectedCells.length) {
                    showError('Select at least one cell before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'storyboard_logic':
                answerData.answer = storyboardOrder;
                if (!storyboardOrder || storyboardOrder.length === 0) {
                    showError('Please arrange the images before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'jigsaw_puzzle':
                // Allow empty submissions - backend will mark as incorrect
                answerData.answer = jigsawPlacements || [];
                // Debug logging
                console.log('Jigsaw placements being submitted:', JSON.stringify(jigsawPlacements, null, 2));
                break;
            default:
                answerData.answer = userAnswerInput.value.trim();
                break;
        }

        answerData.elapsed_time = ((Date.now() - (puzzleStartTime || Date.now())) / 1000).toFixed(2);

        if (submitBtn.style.display !== 'none') {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
        }

        fetch('/api/check_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(answerData)
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Debug logging for jigsaw puzzles
                if (currentPuzzle && currentPuzzle.input_type === 'jigsaw_puzzle') {
                    console.log('Jigsaw validation response:', data);
                    if (!data.correct && data.details) {
                        console.log('Validation details:', data.details);
                    }
                }

                benchmarkStats.total += 1;
                if (data.correct) {
                    benchmarkStats.correct += 1;
                    resultMessage.textContent = 'Correct!';
                    resultMessage.className = 'result-message correct';
                    createFireworks();
                } else {
                    const errorMsg = data.correct_answer ? `Incorrect. ${data.correct_answer}` : 'Incorrect.';
                    resultMessage.textContent = errorMsg;
                    resultMessage.className = 'result-message incorrect';
                    createSadFace();
                }

                updateStats();
                recordBenchmarkResult({
                    puzzle_type: currentPuzzle.puzzle_type,
                    puzzle_id: currentPuzzle.puzzle_id,
                    user_answer: answerData.answer,
                    correct_answer: data.correct_answer,
                    correct: data.correct,
                    elapsed_time: answerData.elapsed_time
                });

                // Reset custom submit buttons (including jigsaw) before loading new puzzle
                resetCustomSubmitButtons();
                
                setTimeout(loadNewPuzzle, 2000);
            })
            .catch((error) => {
                console.error('Error checking answer:', error);
                showError('Error checking answer. Please try again.');
                if (submitBtn.style.display !== 'none') {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Submit';
                }
                resetCustomSubmitButtons();
            });
    }

    function resetCustomSubmitButtons() {
        const bingoButton = document.querySelector('.submit-bingo');
        if (bingoButton) {
            bingoButton.disabled = false;
            bingoButton.textContent = 'Swap and Submit';
        }

        const shadowButton = document.querySelector('.submit-shadow');
        if (shadowButton) {
            shadowButton.disabled = false;
            shadowButton.textContent = 'Submit';
        }

        const mirrorButton = document.querySelector('.submit-mirror');
        if (mirrorButton) {
            mirrorButton.disabled = false;
            mirrorButton.textContent = 'Submit';
        }

        const squiggleButton = document.querySelector('.submit-squiggle');
        if (squiggleButton) {
            squiggleButton.disabled = false;
            squiggleButton.textContent = 'Submit';
        }

        const transformPipelineButton = document.querySelector('.submit-transform-pipeline');
        if (transformPipelineButton) {
            transformPipelineButton.disabled = false;
            transformPipelineButton.textContent = 'Submit';
        }

        const storyboardButton = document.querySelector('.submit-storyboard');
        if (storyboardButton) {
            storyboardButton.disabled = false;
            storyboardButton.textContent = 'Submit Order';
        }

        const jigsawButton = document.querySelector('.submit-jigsaw');
        if (jigsawButton) {
            jigsawButton.disabled = false;
            jigsawButton.textContent = 'Submit Puzzle';
        }
    }

    function updateStats() {
        totalCount.textContent = benchmarkStats.total;
        correctCount.textContent = benchmarkStats.correct;

        const accuracy = benchmarkStats.total
            ? ((benchmarkStats.correct / benchmarkStats.total) * 100).toFixed(1)
            : '0.0';
        accuracyEl.textContent = `${accuracy}%`;
    }

    function recordBenchmarkResult(result) {
        if (!result.timestamp) {
            result.timestamp = new Date().toISOString();
        }

        fetch('/api/benchmark_results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(result)
        }).catch((error) => {
            console.error('Error recording benchmark result:', error);
        });
    }

    function displayDifficultyStars(puzzleType) {
        const difficultyRatings = {
            Dice_Count: 3,
            Bingo: 3,
            Shadow_Plausible: 4,
            Mirror: 4,
            Squiggle: 4,
            Color_Cipher: 3,
            Red_Dot: 4,
            Storyboard_Logic: 3,
            Jigsaw_Puzzle: 2,
            Transform_Pipeline: 4,
        };

        const difficulty = difficultyRatings[puzzleType] || 1;
        const starsContainer = document.getElementById('difficulty-stars');
        if (!starsContainer) {
            return;
        }

        starsContainer.innerHTML = '';
        for (let i = 0; i < 5; i += 1) {
            const star = document.createElement('span');
            star.className = 'star';
            star.innerHTML = i < difficulty ? '★' : '☆';
            starsContainer.appendChild(star);
        }
    }

    function showError(message) {
        resultMessage.textContent = message;
        resultMessage.className = 'result-message incorrect';
    }

    function createFireworks() {
        const container = document.createElement('div');
        container.className = 'fireworks-container';

        const burstCount = 6;
        const sparkCount = 12;

        for (let burstIndex = 0; burstIndex < burstCount; burstIndex += 1) {
            const burst = document.createElement('div');
            burst.className = 'firework-burst';

            const topPercent = Math.random() * 70 + 10;
            const leftPercent = Math.random() * 80 + 10;
            burst.style.top = `${topPercent}%`;
            burst.style.left = `${leftPercent}%`;

            for (let sparkIndex = 0; sparkIndex < sparkCount; sparkIndex += 1) {
                const spark = document.createElement('span');
                spark.className = 'firework-spark';

                const hue = Math.floor(Math.random() * 360);
                spark.style.background = `radial-gradient(circle, hsl(${hue}, 100%, 70%) 0%, hsl(${hue}, 100%, 50%) 60%)`;

                spark.style.setProperty('--spark-index', sparkIndex);
                const delay = (burstIndex * 0.12) + (sparkIndex * 0.03);
                spark.style.animationDelay = `${delay}s`;

                burst.appendChild(spark);
            }

            container.appendChild(burst);
        }

        document.body.appendChild(container);

        setTimeout(() => {
            container.remove();
        }, 1600);
    }

    function createSadFace() {
        const container = document.createElement('div');
        container.className = 'sad-face-container';
        container.textContent = '😞';

        document.body.appendChild(container);

        setTimeout(() => {
            container.remove();
        }, 1500);
    }
});
