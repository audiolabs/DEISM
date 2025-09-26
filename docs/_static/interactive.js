// Interactive JavaScript for DEISM parameter documentation

document.addEventListener('DOMContentLoaded', function() {
    console.log('Interactive.js loaded');
    
    // Configuration for popup content
    const popupConfig = {
        showDescription: true,
        showParameters: true,
        showReturns: true,
        showUsesParameters: true,
        showComputedParameters: true,
        showAlgorithmDetails: true,
        showExamples: true,
        maxWidth: '600px',
        maxHeight: '500px'
    };
    
    // Search functionality
    const searchInput = document.querySelector('.param-search');
    const paramCards = document.querySelectorAll('.param-card');
    
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            
            paramCards.forEach(card => {
                const title = card.querySelector('.param-title').textContent.toLowerCase();
                const description = card.querySelector('.param-description').textContent.toLowerCase();
                const category = card.querySelector('.param-category').textContent.toLowerCase();
                
                if (title.includes(searchTerm) || description.includes(searchTerm) || category.includes(searchTerm)) {
                    card.style.display = 'block';
                    if (searchTerm) {
                        highlightText(card, searchTerm);
                    }
                } else {
                    card.style.display = 'none';
                }
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'k') {
                e.preventDefault();
                searchInput.focus();
            }
            if (e.key === 'Escape') {
                searchInput.value = '';
                searchInput.dispatchEvent(new Event('input'));
                searchInput.blur();
            }
        });
    }
    
    // Function popup system
    let functionPopup = null;
    
    function createFunctionPopup() {
        if (functionPopup) return functionPopup;
        
        functionPopup = document.createElement('div');
        functionPopup.className = 'function-popup';
        functionPopup.style.cssText = `
            position: fixed;
            background: white;
            border: 2px solid #343131;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            padding: 20px;
            max-width: ${popupConfig.maxWidth};
            max-height: ${popupConfig.maxHeight};
            overflow-y: auto;
            z-index: 10000;
            display: none;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.5;
        `;
        
        document.body.appendChild(functionPopup);
        return functionPopup;
    }
    
    function showFunctionPopup(functionName, mouseX, mouseY) {
        console.log('Showing popup for function:', functionName);
        const popup = createFunctionPopup();
        
        // Extract function information from references and static data
        const functionInfo = extractFunctionInfoFromReferences(functionName);
        
        if (functionInfo.description || functionInfo.usedInParameters.length > 0) {
            console.log('Found function information:', functionInfo);
            popup.innerHTML = generatePopupHTML(functionName, functionInfo);
        } else {
            console.log('Function information not found, showing basic popup');
            showBasicPopup(popup, functionName);
        }
        
        // Show popup first to get accurate dimensions
        popup.style.display = 'block';
        
        // Position popup intelligently
        positionPopup(popup, mouseX, mouseY);
        
        // Add event handlers
        setupPopupEventHandlers(popup);
    }
    
    function showBasicPopup(popup, functionName) {
        popup.innerHTML = generateBasicPopupHTML(functionName);
    }
    
    function extractComprehensiveFunctionInfo(element, functionName) {
        const info = {
            description: '',
            parameters: [],
            returns: '',
            usesParameters: [],
            computedParameters: [],
            algorithmDetails: '',
            examples: '',
            notes: ''
        };
        
        // Get the function signature and description
        const descElement = element.querySelector('p');
        if (descElement && popupConfig.showDescription) {
            info.description = descElement.textContent.trim();
        }
        
        // Extract parameters from Sphinx field lists
        if (popupConfig.showParameters) {
            const fieldLists = element.querySelectorAll('.field-list');
            fieldLists.forEach(fieldList => {
                const fields = fieldList.querySelectorAll('.field');
                fields.forEach(field => {
                    const fieldName = field.querySelector('.field-name');
                    const fieldBody = field.querySelector('.field-body');
                    if (fieldName && fieldBody) {
                        const name = fieldName.textContent.trim().toLowerCase();
                        const content = fieldBody.textContent.trim();
                        
                        if (name.includes('param')) {
                            const parts = content.split(':');
                            if (parts.length >= 2) {
                                info.parameters.push({
                                    name: parts[0].trim(),
                                    type: parts[1] ? parts[1].trim() : '',
                                    description: parts.slice(2).join(':').trim()
                                });
                            }
                        } else if (name.includes('return')) {
                            info.returns = content;
                        } else if (name.includes('rtype')) {
                            info.returnType = content;
                        }
                    }
                });
            });
        }
        
        // Extract "Uses Parameters" section
        if (popupConfig.showUsesParameters) {
            const usesParamsSection = findSectionContent(element, ['Uses Parameters', 'Key Parameters Used', 'Input Parameters']);
            if (usesParamsSection) {
                info.usesParameters = extractParameterList(usesParamsSection);
            }
        }
        
        // Extract "Computed Parameters" section
        if (popupConfig.showComputedParameters) {
            const computedParamsSection = findSectionContent(element, ['Computed Parameters', 'Output Parameters']);
            if (computedParamsSection) {
                info.computedParameters = extractParameterList(computedParamsSection);
            }
        }
        
        // Extract "Algorithm Selection" or similar sections
        if (popupConfig.showAlgorithmDetails) {
            const algorithmSection = findSectionContent(element, ['Algorithm Selection', 'Algorithm Details', 'Main Methods']);
            if (algorithmSection) {
                info.algorithmDetails = algorithmSection.textContent.trim();
            }
        }
        
        // Extract any notes or additional information
        const noteElements = element.querySelectorAll('div.admonition, .note, .warning');
        if (noteElements.length > 0) {
            info.notes = Array.from(noteElements).map(note => note.textContent.trim()).join('\n\n');
        }
        
        return info;
    }
    
    function findSectionContent(element, sectionTitles) {
        for (const title of sectionTitles) {
            // Look for strong/bold elements containing the section title
            const strongElements = element.querySelectorAll('strong');
            for (const strong of strongElements) {
                if (strong.textContent.includes(title)) {
                    // Get the next sibling element(s) which should contain the content
                    let nextElement = strong.parentElement.nextElementSibling;
                    if (!nextElement) {
                        nextElement = strong.nextElementSibling;
                    }
                    if (nextElement) {
                        return nextElement;
                    }
                }
            }
        }
        return null;
    }
    
    function extractParameterList(element) {
        const parameters = [];
        const listItems = element.querySelectorAll('li');
        listItems.forEach(item => {
            const text = item.textContent.trim();
            const parts = text.split(' - ');
            if (parts.length >= 2) {
                parameters.push({
                    name: parts[0].trim(),
                    description: parts.slice(1).join(' - ').trim()
                });
            }
        });
        return parameters;
    }
    
    function extractFunctionInfoFromReferences(functionName) {
        const info = {
            description: '',
            parameters: [],
            returns: '',
            usedInParameters: [],
            computedParameters: [],
            algorithmDetails: '',
            examples: '',
            notes: ''
        };
        
        // Static function descriptions database
        const functionDescriptions = {
            'compute_rest_params': 'Computes derived parameters from input configuration. Calculates frequency arrays, wavenumbers, impedance matrices, and other derived values needed for DEISM calculations.',
            'detect_conflicts': 'Validates parameter consistency and detects potential conflicts. Checks source/receiver types, distances, and other parameter relationships.',
            'run_DEISM': 'Main DEISM algorithm entry point. Selects appropriate algorithm variant (ORG, LC, or MIX) and executes room impulse response calculation.',
            'pre_calc_images_src_rec_original': 'Generates image sources using original DEISM method. Creates all image source positions and calculates distances for reflection orders up to maxReflOrder.',
            'pre_calc_images_src_rec_lowComplexity': 'Generates image sources using low-complexity DEISM method. Faster approximation suitable for higher-order reflections.',
            'pre_calc_images_src_rec_MIX': 'Generates image sources for MIX mode. Separates early and late reflections based on mixEarlyOrder parameter.',
            'ray_run_DEISM': 'Executes DEISM calculation for a batch of image sources. Processes numParaImages at a time for memory efficiency.',
            'ray_run_DEISM_LC': 'Low-complexity DEISM calculation for image source batches. Vectorized implementation for speed.',
            'ray_run_DEISM_MIX': 'Mixed-mode DEISM calculation combining ORG and LC methods based on reflection order.',
            'init_source_directivity': 'Initializes source directivity patterns. Loads analytical or measured directivity data for spherical harmonic expansion.',
            'init_receiver_directivity': 'Initializes receiver directivity patterns. Handles normalization and spherical harmonic coefficients.',
            'run_DEISM_ARG': 'DEISM-ARG variant for arbitrary room geometries. Extends DEISM to non-rectangular rooms.',
            'load_directive_pressure': 'Loads sampled directivity pressure fields from COMSOL simulations or measurements.',
            'load_RTF_data': 'Loads room transfer function data from COMSOL simulations for validation.',
            'readYaml': 'Reads YAML configuration files and converts parameters to appropriate data types.',
            'parseCmdArgs': 'Parses command line arguments for parameter override functionality.',
            'loadSingleParam': 'Loads parameters from YAML configuration with command line overrides.',
            'printDict': 'Prints formatted parameter summary with key information and usage notes.'
        };
        
        // Get basic description from static database
        if (functionDescriptions[functionName]) {
            info.description = functionDescriptions[functionName];
        }
        
        // Find all function references in the document
        const functionRefs = document.querySelectorAll(`a.function-ref[href="#${functionName}"]`);
        
        // Extract information from each reference
        functionRefs.forEach(ref => {
            // Get the title attribute (brief description)
            const title = ref.getAttribute('title');
            if (title && title.trim()) {
                // Use the title as algorithm details if we don't have a description
                if (!info.description) {
                    info.description = title;
                }
                // Add to algorithm details
                if (info.algorithmDetails) {
                    info.algorithmDetails += '\n• ' + title;
                } else {
                    info.algorithmDetails = '• ' + title;
                }
            }
            
            // Find the parameter this function is used in
            const paramCard = ref.closest('.param-card');
            if (paramCard) {
                const paramTitle = paramCard.querySelector('.param-title');
                const paramDescription = paramCard.querySelector('.param-description');
                const paramCategory = paramCard.querySelector('.param-category');
                
                if (paramTitle) {
                    const paramName = paramTitle.textContent.trim();
                    const paramDesc = paramDescription ? paramDescription.textContent.trim() : '';
                    const paramCat = paramCategory ? paramCategory.textContent.trim() : '';
                    
                    info.usedInParameters.push({
                        name: paramName,
                        description: paramDesc,
                        category: paramCat,
                        usage: title || 'Parameter processing'
                    });
                }
            }
        });
        
        // Add parameter information from static database
        const functionParameters = {
            'compute_rest_params': [
                { name: 'params', type: 'dict', description: 'Configuration parameters dictionary' }
            ],
            'detect_conflicts': [
                { name: 'params', type: 'dict', description: 'Configuration parameters dictionary' }
            ],
            'run_DEISM': [
                { name: 'params', type: 'dict', description: 'Configuration parameters dictionary' },
                { name: 'images', type: 'numpy.ndarray', description: 'Image source positions' },
                { name: 'directivities', type: 'dict', description: 'Source/receiver directivity data' }
            ],
            'readYaml': [
                { name: 'filePath', type: 'str', description: 'Path to YAML configuration file' }
            ],
            'loadSingleParam': [
                { name: 'configs', type: 'dict', description: 'YAML configuration data' },
                { name: 'args', type: 'argparse.Namespace', description: 'Command line arguments' }
            ]
        };
        
        if (functionParameters[functionName]) {
            info.parameters = functionParameters[functionName];
        }
        
        // Add return information
        const functionReturns = {
            'compute_rest_params': 'Updated parameters dictionary with computed values',
            'detect_conflicts': 'None (prints warnings for conflicts)',
            'run_DEISM': 'Room impulse response or transfer function',
            'readYaml': 'Configuration dictionary with numpy arrays',
            'loadSingleParam': 'Final parameters dictionary'
        };
        
        if (functionReturns[functionName]) {
            info.returns = functionReturns[functionName];
        }
        
        return info;
    }
    
    function generatePopupHTML(functionName, info) {
        let html = `
            <div class="popup-header" style="position: relative; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid #e0e0e0;">
                <h4 style="margin: 0; color: #343131; font-size: 18px; font-weight: 600; padding-right: 30px;">
                    <code style="background: #f0f0f0; padding: 4px 8px; border-radius: 4px; font-size: 16px;">${functionName}()</code>
                </h4>
                <button class="popup-close" style="
                    position: absolute;
                    top: -4px;
                    right: 0;
                    background: none;
                    border: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: #666;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    transition: all 0.2s;
                ">&times;</button>
            </div>
            <div class="popup-content" style="color: #555;">
        `;
        
        // Description
        if (info.description) {
            html += `
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; color: #343131; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">📝 Description</h5>
                    <p style="margin: 0; line-height: 1.6;">${info.description}</p>
                </div>
            `;
        }
        
        // Parameters
        if (info.parameters.length > 0) {
            html += `
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; color: #343131; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">⚙️ Parameters</h5>
                    <div style="background: #f8f9fa; padding: 12px; border-radius: 6px; border-left: 4px solid #007bff;">
            `;
            info.parameters.forEach(param => {
                html += `
                    <div style="margin-bottom: 8px;">
                        <code style="background: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 3px; font-size: 12px; font-weight: 600;">${param.name}</code>
                        ${param.type ? `<span style="color: #666; font-size: 12px; margin-left: 8px;">(${param.type})</span>` : ''}
                        ${param.description ? `<div style="margin-top: 4px; font-size: 13px; color: #555;">${param.description}</div>` : ''}
                    </div>
                `;
            });
            html += '</div></div>';
        }
        
        // Returns
        if (info.returns) {
            html += `
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; color: #343131; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">↩️ Returns</h5>
                    <div style="background: #e8f5e8; padding: 12px; border-radius: 6px; border-left: 4px solid #28a745;">
                        <span style="font-size: 13px; color: #555;">${info.returns}</span>
                    </div>
                </div>
            `;
        }
        
        // Uses Parameters
        if (info.usesParameters.length > 0 || info.usedInParameters.length > 0) {
            html += `
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; color: #343131; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">🔗 Uses Parameters</h5>
                    <div style="background: #fff3cd; padding: 12px; border-radius: 6px; border-left: 4px solid #ffc107;">
            `;
            
            // Handle old format (usesParameters)
            info.usesParameters.forEach(param => {
                html += `
                    <div style="margin-bottom: 6px; font-size: 13px;">
                        <code style="background: #ffeaa7; color: #b36400; padding: 2px 4px; border-radius: 3px; font-size: 11px;">${param.name}</code>
                        <span style="color: #555; margin-left: 8px;">${param.description}</span>
                    </div>
                `;
            });
            
            // Handle new format (usedInParameters)
            info.usedInParameters.forEach(param => {
                html += `
                    <div style="margin-bottom: 8px; font-size: 13px; padding: 6px; background: rgba(255, 255, 255, 0.5); border-radius: 4px;">
                        <div style="display: flex; align-items: center; margin-bottom: 4px;">
                            <code style="background: #ffeaa7; color: #b36400; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: 600;">${param.name}</code>
                            <span style="background: #e8f4f8; color: #0277bd; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-left: 8px; text-transform: uppercase;">${param.category}</span>
                        </div>
                        <div style="color: #555; font-size: 12px; margin-bottom: 2px;">${param.description}</div>
                        <div style="color: #777; font-size: 11px; font-style: italic;">Usage: ${param.usage}</div>
                    </div>
                `;
            });
            
            html += '</div></div>';
        }
        
        // Computed Parameters
        if (info.computedParameters.length > 0) {
            html += `
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; color: #343131; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">⚡ Computed Parameters</h5>
                    <div style="background: #f3e5f5; padding: 12px; border-radius: 6px; border-left: 4px solid #9c27b0;">
            `;
            info.computedParameters.forEach(param => {
                html += `
                    <div style="margin-bottom: 6px; font-size: 13px;">
                        <code style="background: #e1bee7; color: #7b1fa2; padding: 2px 4px; border-radius: 3px; font-size: 11px;">${param.name}</code>
                        <span style="color: #555; margin-left: 8px;">${param.description}</span>
                    </div>
                `;
            });
            html += '</div></div>';
        }
        
        // Algorithm Details
        if (info.algorithmDetails) {
            html += `
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; color: #343131; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">🧮 Algorithm Details</h5>
                    <div style="background: #ffe8f0; padding: 12px; border-radius: 6px; border-left: 4px solid #e91e63;">
                        <div style="font-size: 13px; color: #555; white-space: pre-line;">${info.algorithmDetails}</div>
                    </div>
                </div>
            `;
        }
        
        // Notes
        if (info.notes) {
            html += `
                <div style="margin-bottom: 16px;">
                    <h5 style="margin: 0 0 8px 0; color: #343131; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">📌 Notes</h5>
                    <div style="background: #e3f2fd; padding: 12px; border-radius: 6px; border-left: 4px solid #2196f3;">
                        <span style="font-size: 13px; color: #555;">${info.notes}</span>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        return html;
    }
    
    function generateBasicPopupHTML(functionName) {
        return `
            <div class="popup-header" style="position: relative; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid #e0e0e0;">
                <h4 style="margin: 0; color: #343131; font-size: 18px; font-weight: 600; padding-right: 30px;">
                    <code style="background: #f0f0f0; padding: 4px 8px; border-radius: 4px; font-size: 16px;">${functionName}()</code>
                </h4>
                <button class="popup-close" style="
                    position: absolute;
                    top: -4px;
                    right: 0;
                    background: none;
                    border: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: #666;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                ">&times;</button>
            </div>
            <div class="popup-content" style="color: #555;">
                <div style="background: #f8f9fa; padding: 16px; border-radius: 6px; border-left: 4px solid #6c757d;">
                    <p style="margin: 0; font-style: italic;">📖 This function is documented in the API reference above. Scroll up to see the detailed documentation.</p>
                </div>
            </div>
        `;
    }
    
    function positionPopup(popup, mouseX, mouseY) {
        const rect = popup.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        let x = mouseX + 15;
        let y = mouseY + 15;
        
        // Adjust if popup would go off-screen
        if (x + rect.width > viewportWidth - 20) {
            x = mouseX - rect.width - 15;
        }
        if (y + rect.height > viewportHeight - 20) {
            y = mouseY - rect.height - 15;
        }
        
        // Ensure popup stays within viewport
        x = Math.max(10, Math.min(x, viewportWidth - rect.width - 10));
        y = Math.max(10, Math.min(y, viewportHeight - rect.height - 10));
        
        popup.style.left = x + 'px';
        popup.style.top = y + 'px';
    }
    
    function setupPopupEventHandlers(popup) {
        // Close button
        const closeButton = popup.querySelector('.popup-close');
        closeButton.addEventListener('click', function(e) {
            e.stopPropagation();
            hideFunctionPopup();
        });
        
        // Escape key
        const escapeHandler = function(e) {
            if (e.key === 'Escape') {
                hideFunctionPopup();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
        
        // Click outside
        setTimeout(() => {
            const outsideClickHandler = function(e) {
                if (!popup.contains(e.target) && !e.target.closest('.function-ref')) {
                    hideFunctionPopup();
                    document.removeEventListener('click', outsideClickHandler);
                }
            };
            document.addEventListener('click', outsideClickHandler);
        }, 100);
    }
    
    function hideFunctionPopup() {
        console.log('Hiding popup');
        if (functionPopup) {
            functionPopup.style.display = 'none';
        }
    }
    
    // Single event delegation for function reference links (FIXED: no duplicates)
    document.addEventListener('click', function(e) {
        console.log('Click detected on:', e.target);
        
        let target = e.target;
        while (target && target !== document) {
            if (target.classList && target.classList.contains('function-ref')) {
                console.log('Function ref clicked:', target);
                e.preventDefault();
                e.stopPropagation();
                
                const href = target.getAttribute('href');
                if (href && href.startsWith('#')) {
                    const functionName = href.substring(1);
                    showFunctionPopup(functionName, e.clientX, e.clientY);
                    return false;
                }
            }
            target = target.parentElement;
        }
    }, true);
    
    // Tooltip functionality for parameter names
    const paramTitles = document.querySelectorAll('.param-title');
    paramTitles.forEach(title => {
        title.addEventListener('mouseenter', function(e) {
            const card = this.closest('.param-card');
            const usedInElement = card.querySelector('.used-in');
            if (usedInElement) {
                showTooltip(e, usedInElement.textContent);
            }
        });
        
        title.addEventListener('mouseleave', function() {
            hideTooltip();
        });
    });
    
    // Tooltip system
    let tooltip = null;
    
    function showTooltip(event, content) {
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.className = 'param-tooltip';
            tooltip.style.cssText = `
                position: fixed;
                background: rgba(0,0,0,0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 9999;
                max-width: 300px;
                word-wrap: break-word;
                pointer-events: none;
            `;
            document.body.appendChild(tooltip);
        }
        
        tooltip.textContent = content;
        tooltip.style.left = (event.clientX + 10) + 'px';
        tooltip.style.top = (event.clientY - 30) + 'px';
        tooltip.style.display = 'block';
    }
    
    function hideTooltip() {
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }
    
    // Highlight search terms
    function highlightText(element, searchTerm) {
        const existingHighlights = element.querySelectorAll('.search-highlight');
        existingHighlights.forEach(highlight => {
            highlight.outerHTML = highlight.textContent;
        });
        
        if (!searchTerm) return;
        
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }
        
        textNodes.forEach(textNode => {
            const text = textNode.textContent.toLowerCase();
            if (text.includes(searchTerm)) {
                const parent = textNode.parentNode;
                const regex = new RegExp(`(${searchTerm})`, 'gi');
                const highlightedHTML = textNode.textContent.replace(regex, '<span class="search-highlight">$1</span>');
                const wrapper = document.createElement('div');
                wrapper.innerHTML = highlightedHTML;
                while (wrapper.firstChild) {
                    parent.insertBefore(wrapper.firstChild, textNode);
                }
                parent.removeChild(textNode);
            }
        });
    }
    
    // Allow users to configure popup content
    window.configurePopup = function(config) {
        Object.assign(popupConfig, config);
        console.log('Popup configuration updated:', popupConfig);
    };
});

// Search functionality
function initializeSearch() {
    const searchInput = document.querySelector('.param-search');
    const paramCards = document.querySelectorAll('.param-card');
    
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            
            paramCards.forEach(card => {
                const title = card.querySelector('.param-title').textContent.toLowerCase();
                const description = card.querySelector('.param-description').textContent.toLowerCase();
                const category = card.querySelector('.param-category').textContent.toLowerCase();
                
                if (title.includes(searchTerm) || description.includes(searchTerm) || category.includes(searchTerm)) {
                    card.style.display = 'block';
                    // Highlight matching text
                    if (searchTerm) {
                        highlightText(card, searchTerm);
                    }
                } else {
                    card.style.display = 'none';
                }
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'k') {
                e.preventDefault();
                searchInput.focus();
            }
            if (e.key === 'Escape') {
                searchInput.value = '';
                searchInput.dispatchEvent(new Event('input'));
                searchInput.blur();
            }
        });
    }
}

function searchParameters(searchTerm) {
    const paramCards = document.querySelectorAll('.param-card');
    let hasResults = false;
    
    paramCards.forEach(card => {
        const cardText = card.textContent.toLowerCase();
        const paramTitle = card.querySelector('.param-title');
        
        if (cardText.includes(searchTerm) || searchTerm === '') {
            card.style.display = 'block';
            hasResults = true;
            
            // Highlight search terms
            if (searchTerm && paramTitle) {
                highlightSearchTerm(paramTitle, searchTerm);
            }
        } else {
            card.style.display = 'none';
        }
    });
    
    // Show/hide no results message
    showNoResultsMessage(!hasResults && searchTerm);
}

function highlightSearchTerm(element, searchTerm) {
    const text = element.textContent;
    const regex = new RegExp(`(${searchTerm})`, 'gi');
    const highlighted = text.replace(regex, '<span class="search-highlight">$1</span>');
    element.innerHTML = highlighted;
}

function showNoResultsMessage(show) {
    let noResults = document.querySelector('.no-results');
    if (show && !noResults) {
        noResults = document.createElement('div');
        noResults.className = 'no-results';
        noResults.innerHTML = '<p>No parameters found matching your search.</p>';
        document.querySelector('.param-search').parentNode.appendChild(noResults);
    } else if (!show && noResults) {
        noResults.remove();
    }
}

// Collapsible sections
function initializeCollapsibleSections() {
    const sectionHeaders = document.querySelectorAll('h2, h3');
    
    sectionHeaders.forEach(header => {
        if (header.nextElementSibling) {
            header.style.cursor = 'pointer';
            header.addEventListener('click', function() {
                const content = this.nextElementSibling;
                if (content.style.display === 'none') {
                    content.style.display = 'block';
                    this.textContent = this.textContent.replace('▶', '▼');
                } else {
                    content.style.display = 'none';
                    this.textContent = '▶ ' + this.textContent.replace('▼ ', '');
                }
            });
        }
    });
}

// Parameter tooltips
function initializeParameterTooltips() {
    const tooltips = document.querySelectorAll('.param-tooltip');
    
    tooltips.forEach(tooltip => {
        const content = tooltip.querySelector('.tooltip-content');
        if (content) {
            // Position tooltip dynamically
            tooltip.addEventListener('mouseenter', function() {
                positionTooltip(tooltip, content);
            });
            
            // Close tooltip on click outside
            document.addEventListener('click', function(e) {
                if (!tooltip.contains(e.target)) {
                    content.style.visibility = 'hidden';
                    content.style.opacity = '0';
                }
            });
        }
    });
}

function positionTooltip(tooltip, content) {
    const rect = tooltip.getBoundingClientRect();
    const contentRect = content.getBoundingClientRect();
    const viewportHeight = window.innerHeight;
    
    // Check if tooltip would go off-screen
    if (rect.top - contentRect.height < 0) {
        // Show below instead of above
        content.style.bottom = 'auto';
        content.style.top = '125%';
        content.style.transform = 'none';
        
        // Adjust arrow
        const arrow = content.querySelector('::after');
        if (arrow) {
            content.style.setProperty('--arrow-direction', 'up');
        }
    } else {
        // Default position (above)
        content.style.bottom = '125%';
        content.style.top = 'auto';
        content.style.transform = 'none';
    }
}

// Function links
function initializeFunctionLinks() {
    const functionRefs = document.querySelectorAll('.function-ref');
    
    functionRefs.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const functionName = this.textContent.trim();
            navigateToFunction(functionName);
        });
    });
}

function navigateToFunction(functionName) {
    // Try to find the function in the current page
    const functionElement = document.querySelector(`[id*="${functionName}"]`);
    if (functionElement) {
        functionElement.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
        
        // Highlight the function temporarily
        functionElement.classList.add('highlight-function');
        setTimeout(() => {
            functionElement.classList.remove('highlight-function');
        }, 2000);
    } else {
        // Try to navigate to API reference
        const apiUrl = `api_reference.html#${functionName}`;
        window.open(apiUrl, '_blank');
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add smooth scrolling for anchor links
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const targetId = e.target.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        
        if (targetElement) {
            targetElement.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

// Add keyboard navigation
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchBox = document.querySelector('.param-search');
        if (searchBox) {
            searchBox.focus();
        }
    }
    
    // Escape to clear search
    if (e.key === 'Escape') {
        const searchBox = document.querySelector('.param-search');
        if (searchBox && searchBox === document.activeElement) {
            searchBox.value = '';
            searchParameters('');
        }
    }
});

// Add loading animation for dynamic content
function showLoadingSpinner(element) {
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    spinner.innerHTML = '<div class="spinner"></div>';
    element.appendChild(spinner);
}

function hideLoadingSpinner(element) {
    const spinner = element.querySelector('.loading-spinner');
    if (spinner) {
        spinner.remove();
    }
}

// Add copy to clipboard functionality
function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.textContent = 'Copy';
        copyButton.addEventListener('click', function() {
            navigator.clipboard.writeText(block.textContent).then(() => {
                copyButton.textContent = 'Copied!';
                setTimeout(() => {
                    copyButton.textContent = 'Copy';
                }, 2000);
            });
        });
        
        block.parentNode.style.position = 'relative';
        block.parentNode.appendChild(copyButton);
    });
}

// Initialize copy buttons when DOM is ready
document.addEventListener('DOMContentLoaded', addCopyButtons);

// Add print-friendly styles
function addPrintSupport() {
    const printButton = document.createElement('button');
    printButton.className = 'print-button';
    printButton.textContent = 'Print';
    printButton.addEventListener('click', function() {
        window.print();
    });
    
    const toolbar = document.querySelector('.toolbar');
    if (toolbar) {
        toolbar.appendChild(printButton);
    }
}

// Performance optimization: lazy load function references
function lazyLoadFunctionReferences() {
    const functionRefs = document.querySelectorAll('.function-ref[data-lazy]');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                const functionName = element.dataset.function;
                loadFunctionReference(element, functionName);
                observer.unobserve(element);
            }
        });
    });
    
    functionRefs.forEach(ref => observer.observe(ref));
}

function loadFunctionReference(element, functionName) {
    // Simulate loading function documentation
    fetch(`/api/function/${functionName}`)
        .then(response => response.json())
        .then(data => {
            element.title = data.description;
            element.classList.add('loaded');
        })
        .catch(error => {
            console.warn(`Could not load reference for ${functionName}:`, error);
        });
}

// Add accessibility improvements
function improveAccessibility() {
    // Add ARIA labels
    const paramCards = document.querySelectorAll('.param-card');
    paramCards.forEach((card, index) => {
        card.setAttribute('role', 'article');
        card.setAttribute('aria-label', `Parameter ${index + 1}`);
    });
    
    // Add keyboard navigation for tooltips
    const tooltips = document.querySelectorAll('.param-tooltip');
    tooltips.forEach(tooltip => {
        tooltip.setAttribute('tabindex', '0');
        tooltip.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const content = this.querySelector('.tooltip-content');
                if (content) {
                    content.style.visibility = 'visible';
                    content.style.opacity = '1';
                }
            }
        });
    });
}

// Initialize accessibility improvements
document.addEventListener('DOMContentLoaded', improveAccessibility);

// Function popup system
let functionPopup = null;

function createFunctionPopup() {
    if (functionPopup) return functionPopup;
    
    functionPopup = document.createElement('div');
    functionPopup.className = 'function-popup';
    functionPopup.style.cssText = `
        position: fixed;
        background: white;
        border: 2px solid #343131;
        border-radius: 8px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        padding: 16px;
        max-width: 500px;
        max-height: 400px;
        overflow-y: auto;
        z-index: 10000;
        display: none;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        line-height: 1.4;
    `;
    
    document.body.appendChild(functionPopup);
    return functionPopup;
}

function showFunctionPopup(functionName, mouseX, mouseY) {
    console.log('Showing popup for function:', functionName);
    const popup = createFunctionPopup();
    
    // Find the function documentation
    const functionElement = document.querySelector(`[id="${functionName}"]`);
    if (!functionElement) {
        console.warn(`Function ${functionName} not found`);
        // Show a basic popup even if we can't find the function
        popup.innerHTML = `
            <div class="popup-header" style="position: relative;">
                <h4 style="margin: 0 0 8px 0; color: #343131; font-size: 16px; padding-right: 30px;">${functionName}</h4>
                <button class="popup-close" style="
                    position: absolute;
                    top: -4px;
                    right: 0;
                    background: none;
                    border: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: #666;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">&times;</button>
            </div>
            <div class="popup-content">
                <div>This function is documented in the API reference above.</div>
            </div>
        `;
    } else {
        // Extract function information
        const functionInfo = extractFunctionInfo(functionElement);
        
        // Populate popup content
        popup.innerHTML = `
            <div class="popup-header" style="position: relative;">
                <h4 style="margin: 0 0 8px 0; color: #343131; font-size: 16px; padding-right: 30px;">${functionName}</h4>
                <button class="popup-close" style="
                    position: absolute;
                    top: -4px;
                    right: 0;
                    background: none;
                    border: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: #666;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">&times;</button>
            </div>
            <div class="popup-content">
                ${functionInfo.description}
                ${functionInfo.parameters}
                ${functionInfo.returns}
            </div>
        `;
    }
    
    // Show popup first to get accurate dimensions
    popup.style.display = 'block';
    
    // Position popup near mouse, ensuring it stays within viewport
    const rect = popup.getBoundingClientRect();
    const x = Math.min(mouseX + 10, window.innerWidth - rect.width - 20);
    const y = Math.min(mouseY + 10, window.innerHeight - rect.height - 20);
    
    popup.style.left = Math.max(10, x) + 'px';
    popup.style.top = Math.max(10, y) + 'px';
    
    // Add close button functionality
    const closeButton = popup.querySelector('.popup-close');
    closeButton.addEventListener('click', function(e) {
        e.stopPropagation();
        hideFunctionPopup();
    });
    
    // Close on escape key
    const escapeHandler = function(e) {
        if (e.key === 'Escape') {
            hideFunctionPopup();
            document.removeEventListener('keydown', escapeHandler);
        }
    };
    document.addEventListener('keydown', escapeHandler);
    
    // Close when clicking outside
    setTimeout(() => {
        const outsideClickHandler = function(e) {
            if (!popup.contains(e.target) && !e.target.closest('.function-ref')) {
                hideFunctionPopup();
                document.removeEventListener('click', outsideClickHandler);
            }
        };
        document.addEventListener('click', outsideClickHandler);
    }, 100);
}

function hideFunctionPopup() {
    console.log('Hiding popup');
    if (functionPopup) {
        functionPopup.style.display = 'none';
    }
}

function extractFunctionInfo(element) {
    const info = {
        description: '',
        parameters: '',
        returns: ''
    };
    
    // Get description (first paragraph after the function signature)
    const descriptionElement = element.querySelector('p');
    if (descriptionElement) {
        info.description = `<div style="margin-bottom: 12px;"><strong>Description:</strong><br>${descriptionElement.textContent}</div>`;
    }
    
    // Get parameters from Sphinx documentation
    const paramFields = element.querySelectorAll('.field-list .field');
    if (paramFields.length > 0) {
        info.parameters = '<div style="margin-bottom: 12px;"><strong>Parameters:</strong><ul style="margin: 4px 0; padding-left: 20px;">';
        paramFields.forEach(field => {
            const fieldName = field.querySelector('.field-name');
            const fieldBody = field.querySelector('.field-body');
            if (fieldName && fieldBody) {
                const name = fieldName.textContent.trim();
                const desc = fieldBody.textContent.trim();
                if (name.toLowerCase().includes('param')) {
                    info.parameters += `<li style="margin: 4px 0;"><code style="background: #f5f5f5; padding: 2px 4px; border-radius: 3px;">${desc.split(' ')[0]}</code>: ${desc.split(' ').slice(1).join(' ')}</li>`;
                }
            }
        });
        info.parameters += '</ul></div>';
    }
    
    // Get return info
    const returnField = element.querySelector('.field-name');
    if (returnField && returnField.textContent.includes('Returns')) {
        const returnBody = returnField.nextElementSibling;
        if (returnBody) {
            info.returns = `<div><strong>Returns:</strong><br>${returnBody.textContent}</div>`;
        }
    }
    
    // If no detailed info found, get basic info from the element
    if (!info.description && !info.parameters && !info.returns) {
        const allText = element.textContent.trim();
        const lines = allText.split('\n').filter(line => line.trim());
        if (lines.length > 0) {
            // Take first few meaningful lines
            const descLines = lines.slice(0, 5).join(' ').trim();
            info.description = `<div>${descLines}</div>`;
        }
    }
    
    return info;
} 