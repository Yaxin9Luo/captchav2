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
    let deformationSelectedIndex = null;
    let squiggleSelectedIndex = null;
    let squiggleRevealTimeout = null;

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
        deformationSelectedIndex = null;
        squiggleSelectedIndex = null;
        if (squiggleRevealTimeout) {
            clearTimeout(squiggleRevealTimeout);
            squiggleRevealTimeout = null;
        }

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

        puzzleImage.style.display = 'none';

        const customSelectors = [
            '.bingo-grid',
            '.bingo-submit',
            '.shadow-plausible-grid',
            '.shadow-submit',
            '.mirror-layout',
            '.mirror-submit',
            '.deformation-layout',
            '.deformation-submit',
            '.squiggle-preview',
            '.squiggle-options-grid',
            '.squiggle-submit'
        ];

        customSelectors.forEach((selector) => {
            document.querySelectorAll(selector).forEach((element) => element.remove());
        });
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
            case 'deformation_select':
                setupDeformationSelect(data);
                break;
            case 'squiggle_select':
                setupSquiggleSelect(data);
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

        puzzleImage.src = data.image_path;
        puzzleImage.alt = 'CAPTCHA Puzzle';
        puzzleImage.style.display = 'block';
        puzzleImageContainer.appendChild(puzzleImage);
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

        if (data.image_path) {
            puzzleImage.src = data.image_path;
            puzzleImage.alt = 'CAPTCHA Puzzle';
            puzzleImage.style.display = 'block';
            puzzleImageContainer.appendChild(puzzleImage);
        }
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

    function setupDeformationSelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        deformationSelectedIndex = null;

        const layout = document.createElement('div');
        layout.className = 'deformation-layout';

        const referenceSection = document.createElement('div');
        referenceSection.className = 'deformation-reference';

        const referenceLabel = document.createElement('div');
        referenceLabel.className = 'deformation-reference-label';
        referenceLabel.textContent = 'Reference';
        referenceSection.appendChild(referenceLabel);

        const referenceImg = document.createElement('img');
        referenceImg.src = data.reference_image;
        referenceImg.alt = 'Reference deformation setup';
        referenceImg.draggable = false;
        referenceSection.appendChild(referenceImg);

        const optionsSection = document.createElement('div');
        optionsSection.className = 'deformation-options';

        const optionsLabel = document.createElement('div');
        optionsLabel.className = 'deformation-options-label';
        optionsLabel.textContent = 'Select the correct deformation';
        optionsSection.appendChild(optionsLabel);

        const optionsGrid = document.createElement('div');
        optionsGrid.className = 'deformation-options-grid';

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No deformation options available.');
            return;
        }

        const gridSize = data.grid_size || [2, Math.ceil(optionImages.length / 2)];
        const cols = gridSize[1] || optionImages.length || 1;
        optionsGrid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;

        optionImages.forEach((src, index) => {
            const cell = document.createElement('div');
            cell.className = 'deformation-option';
            cell.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Deformation option ${index + 1}`;
            img.draggable = false;
            cell.appendChild(img);

            const overlay = document.createElement('div');
            overlay.className = 'deformation-overlay';
            cell.appendChild(overlay);

            const badge = document.createElement('div');
            badge.className = 'deformation-checkmark';
            badge.textContent = '✓';
            cell.appendChild(badge);

            cell.addEventListener('click', () => selectDeformationOption(index, cell));

            optionsGrid.appendChild(cell);
        });

        optionsSection.appendChild(optionsGrid);
        layout.appendChild(referenceSection);
        layout.appendChild(optionsSection);
        puzzleImageContainer.appendChild(layout);

        const submitSection = document.createElement('div');
        submitSection.className = 'deformation-submit';

        const deformationSubmitBtn = document.createElement('button');
        deformationSubmitBtn.textContent = 'Submit';
        deformationSubmitBtn.className = 'submit-deformation';
        deformationSubmitBtn.type = 'button';
        deformationSubmitBtn.addEventListener('click', () => {
            if (deformationSelectedIndex === null) {
                showError('Select one deformation before submitting.');
                return;
            }
            deformationSubmitBtn.disabled = true;
            deformationSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(deformationSubmitBtn);
        puzzleImageContainer.appendChild(submitSection);
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

        const gridSize = Array.isArray(data.grid_size) ? data.grid_size : null;
        if (gridSize && gridSize.length > 1 && Number.isFinite(gridSize[1]) && gridSize[1] > 0) {
            optionsGrid.style.gridTemplateColumns = `repeat(${gridSize[1]}, minmax(0, 1fr))`;
        } else if (optionImages.length === 4) {
            optionsGrid.style.gridTemplateColumns = 'repeat(2, minmax(0, 1fr))';
        }

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

    function selectDeformationOption(index, cellElement) {
        if (deformationSelectedIndex === index) {
            deformationSelectedIndex = null;
            const overlay = cellElement.querySelector('.deformation-overlay');
            const badge = cellElement.querySelector('.deformation-checkmark');
            if (overlay) {
                overlay.classList.remove('active');
            }
            if (badge) {
                badge.classList.remove('active');
            }
            cellElement.classList.remove('active');
            return;
        }

        const previouslyActive = document.querySelector('.deformation-option.active');
        if (previouslyActive) {
            previouslyActive.classList.remove('active');
            const previousOverlay = previouslyActive.querySelector('.deformation-overlay');
            const previousBadge = previouslyActive.querySelector('.deformation-checkmark');
            if (previousOverlay) {
                previousOverlay.classList.remove('active');
            }
            if (previousBadge) {
                previousBadge.classList.remove('active');
            }
        }

        deformationSelectedIndex = index;
        const overlay = cellElement.querySelector('.deformation-overlay');
        const badge = cellElement.querySelector('.deformation-checkmark');
        if (overlay) {
            overlay.classList.add('active');
        }
        if (badge) {
            badge.classList.add('active');
        }
        cellElement.classList.add('active');
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

    function submitAnswer() {
        if (!currentPuzzle) {
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
            case 'deformation_select':
                answerData.answer = deformationSelectedIndex;
                if (deformationSelectedIndex === null) {
                    showError('Select one deformation before submitting.');
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

                benchmarkStats.total += 1;
                if (data.correct) {
                    benchmarkStats.correct += 1;
                    resultMessage.textContent = 'Correct!';
                    resultMessage.className = 'result-message correct';
                    createFireworks();
                } else {
                    resultMessage.textContent = 'Incorrect.';
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

        const deformationButton = document.querySelector('.submit-deformation');
        if (deformationButton) {
            deformationButton.disabled = false;
            deformationButton.textContent = 'Submit';
        }

        const squiggleButton = document.querySelector('.submit-squiggle');
        if (squiggleButton) {
            squiggleButton.disabled = false;
            squiggleButton.textContent = 'Submit';
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
            Deformation: 4,
            Squiggle: 4
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
