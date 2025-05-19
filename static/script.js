document.addEventListener('DOMContentLoaded', async function() {
    const boxesInputArea = document.getElementById('boxes-input-area');
    const addBoxBtn = document.getElementById('add-box-btn');
    const loadSampleBoxesBtn = document.getElementById('load-sample-boxes-btn');
    const solveBtn = document.getElementById('solve-btn');
    const visualizationsArea = document.getElementById('visualizations-area');
    const metricsSummaryDiv = document.getElementById('metrics-summary');
    const layerMetricsDetailsDiv = document.getElementById('layer-metrics-details');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessageDiv = document.getElementById('error-message');
    
    // New elements for truck selection
    const truckSelector = document.getElementById('truck-selector');
    const containerDcInput = document.getElementById('container-dc');
    const containerWcInput = document.getElementById('container-wc');
    const containerHcInput = document.getElementById('container-hc');

    let boxIdCounter = 1;
    let allProducts = [];
    let allTrucks = [];

    // API Docs Elements
    const apiDocsOpenBtn = document.getElementById('api-docs-open-btn');
    const apiDocsPanel = document.getElementById('api-docs-panel');
    const apiDocsCloseBtn = document.getElementById('api-docs-close-btn');
    const apiDocsContentArea = document.getElementById('api-docs-content-area');


    async function loadProducts() {
        try {
            const response = await fetch('/static/data/Items data.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            allProducts = await response.json();
            console.log('Products loaded:', allProducts.length);
        } catch (error) {
            console.error("Could not load products:", error);
            errorMessageDiv.textContent = `Error loading product data. Autocomplete might not work as expected. Details: ${error.message}`;
            errorMessageDiv.style.display = 'block';
        }
    }

    async function loadTrucks() {
        try {
            const response = await fetch('/static/Data/camions.json'); // Path to your truck data
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            allTrucks = await response.json();
            console.log('Trucks loaded:', allTrucks.length);
            populateTruckSelector();
        } catch (error) {
            console.error("Could not load trucks:", error);
            let currentError = errorMessageDiv.textContent;
            errorMessageDiv.textContent = (currentError ? currentError + "\n" : "") + `Error loading truck data. Truck selection might not work. Details: ${error.message}`;
            errorMessageDiv.style.display = 'block';
        }
    }

    function populateTruckSelector() {
        if (!truckSelector) return;
        allTrucks.forEach((truck, index) => {
            const option = document.createElement('option');
            option.value = index; // Use index to easily retrieve from allTrucks array
            option.textContent = truck.Nom;
            truckSelector.appendChild(option);
        });

        truckSelector.addEventListener('change', handleTruckSelection);
    }

    function addBoxEntry(boxData = {}) {
        const currentId = boxData.id || boxIdCounter; // This ID is used for initial setup, will be managed by updateBoxIds
        const entryDiv = document.createElement('div');
        entryDiv.classList.add('box-entry');
        entryDiv.innerHTML = `
            <label>Product:</label>
            <div class="product-input-container">
                <input type="text" class="box-product-name" placeholder="Type product name..." value="${boxData.productName || ''}" data-internal-id="${currentId}">
                <div class="product-suggestions"></div>
            </div>
            <label>D:</label><input type="number" step="any" class="box-db" placeholder="db" value="${boxData.db || ''}">
            <label>W:</label><input type="number" step="any" class="box-wb" placeholder="wb" value="${boxData.wb || ''}">
            <label>H:</label><input type="number" step="any" class="box-hb" placeholder="hb" value="${boxData.hb || ''}">
            <div class="orientations-container">
                <div class="orientations-header">
                    <label class="orientations-main-label">Allowed Orientations:</label>
                    <span class="orientations-toggle-arrow">▼</span>
                </div>
                <div class="orientations-checkboxes" style="display: none;"> <!-- Initially hidden -->
                    ${[1, 2, 3, 4, 5, 6].map(o => `
                        <div class="orientation-item">
                            <input type="checkbox" id="box-${currentId}-orient-${o}" class="box-orientation" value="${o}" ${ (boxData.allowed_orientations ? boxData.allowed_orientations.includes(o) : true) ? 'checked' : ''}>
                            <label for="box-${currentId}-orient-${o}">${o}</label>
                        </div>
                    `).join('')}
                </div>
            </div>
            <button type="button" class="duplicate-box-btn">Duplicate</button>
            <button type="button" class="remove-box-btn">Remove</button>
        `;
        boxesInputArea.appendChild(entryDiv);

        entryDiv.querySelector('.remove-box-btn').addEventListener('click', function() {
            entryDiv.remove();
            updateBoxIds(); 
        });

        entryDiv.querySelector('.duplicate-box-btn').addEventListener('click', function() {
            const parentBoxEntry = this.closest('.box-entry');
            const productName = parentBoxEntry.querySelector('.box-product-name').value;
            const db = parentBoxEntry.querySelector('.box-db').value;
            const wb = parentBoxEntry.querySelector('.box-wb').value;
            const hb = parentBoxEntry.querySelector('.box-hb').value;
            
            const allowed_orientations = [];
            parentBoxEntry.querySelectorAll('.box-orientation:checked').forEach(checkbox => {
                allowed_orientations.push(parseInt(checkbox.value));
            });

            // Create a new box entry with the duplicated data
            // Do not pass an 'id' so addBoxEntry treats it as a new box
            addBoxEntry({ productName, db, wb, hb, allowed_orientations });
            updateBoxIds(); // Re-sequence IDs after adding the new duplicated box
        });

        // Add toggle functionality for orientations
        const orientationsHeader = entryDiv.querySelector('.orientations-header');
        const orientationsCheckboxes = entryDiv.querySelector('.orientations-checkboxes');
        const toggleArrow = entryDiv.querySelector('.orientations-toggle-arrow');

        if (orientationsHeader && orientationsCheckboxes && toggleArrow) {
            orientationsHeader.addEventListener('click', () => {
                const isHidden = orientationsCheckboxes.style.display === 'none';
                orientationsCheckboxes.style.display = isHidden ? 'flex' : 'none'; // Use flex to arrange items
                toggleArrow.textContent = isHidden ? '▲' : '▼'; // Change arrow direction
            });
        }

        // Default check for new boxes (if not from sample data with specific empty allowed_orientations)
        if (!boxData.id && (!boxData.allowed_orientations || boxData.allowed_orientations.length === 0)) {
            entryDiv.querySelectorAll('.box-orientation').forEach(cb => cb.checked = true);
        }
        
        // Autocomplete for product name
        const productNameInput = entryDiv.querySelector('.box-product-name');
        const suggestionsDiv = entryDiv.querySelector('.product-suggestions');
        const dbInput = entryDiv.querySelector('.box-db');
        const wbInput = entryDiv.querySelector('.box-wb');
        const hbInput = entryDiv.querySelector('.box-hb');

        productNameInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            suggestionsDiv.innerHTML = '';
            if (query.length < 2) { // Only show suggestions if 2+ chars
                suggestionsDiv.style.display = 'none';
                return;
            }

            const filteredProducts = allProducts.filter(p => p.Produit && p.Produit.toLowerCase().includes(query)).slice(0, 10); // Limit suggestions

            if (filteredProducts.length > 0) {
                filteredProducts.forEach(product => {
                    const suggestionItem = document.createElement('div');
                    suggestionItem.textContent = product.Produit;
                    suggestionItem.addEventListener('click', () => {
                        productNameInput.value = product.Produit;
                        dbInput.value = (parseFloat(product["Longueur (cm)"]) * 0.01).toFixed(3);
                        wbInput.value = (parseFloat(product["Largeur (cm)"]) * 0.01).toFixed(3);
                        hbInput.value = (parseFloat(product["Profondeur (cm)"]) * 0.01).toFixed(3); // Corrected: Profondeur for H
                        suggestionsDiv.innerHTML = '';
                        suggestionsDiv.style.display = 'none';
                    });
                    suggestionsDiv.appendChild(suggestionItem);
                });
                suggestionsDiv.style.display = 'block';
            } else {
                suggestionsDiv.style.display = 'none';
            }
        });

        // Hide suggestions when clicking outside
        document.addEventListener('click', function(event) {
            // Check if the click is outside the current product input and its suggestions
            const currentProductInputContainer = productNameInput.closest('.product-input-container');
            if (currentProductInputContainer && !currentProductInputContainer.contains(event.target)) {
                 suggestionsDiv.style.display = 'none';
            }
        });

        if (!boxData.id) { // Only increment if it's a new box, not one from sample data 
            boxIdCounter++;
        }
    }
    
    function updateBoxIds() {
        let newId = 1;
        document.querySelectorAll('.box-entry').forEach(entry => {
            const productNameInput = entry.querySelector('.box-product-name');
            if (productNameInput) {
                productNameInput.dataset.internalId = newId;
            }
    
            // Update orientation checkbox IDs and their labels
            entry.querySelectorAll('.orientation-item').forEach(item => {
                const checkbox = item.querySelector('.box-orientation');
                const label = item.querySelector('label');
                if (checkbox && label) {
                    const orientationValue = checkbox.value; // 1-6
                    const newCheckboxId = `box-${newId}-orient-${orientationValue}`;
                    checkbox.id = newCheckboxId;
                    label.setAttribute('for', newCheckboxId);
                }
            });
            newId++;
        });
        boxIdCounter = newId; // Update the global counter for the next new box
    }

    function loadSampleBoxes() {
        boxesInputArea.innerHTML = ''; 
        boxIdCounter = 1; 
        const sampleBoxes = [
            { id: 1, productName: "ANKER 322 USB-A TO USB-C CABLE (3FT BRAIDED) black", db: 1.00, wb: 0.02, hb: 0.02, allowed_orientations: [1,2,3,4,5,6] },
            { id: 2, productName: "Batteur à main Cristor – 150W", db: 0.25, wb: 0.15, hb: 0.10, allowed_orientations: [1,3,5] },
            { id: 3, productName: "Tefal Success Moule à tarte 30 cm", db: 0.32, wb: 0.32, hb: 0.05, allowed_orientations: [1,2] },
            { id: 4, productName: "Generic Box A", db: 2, wb: 2, hb: 2 }, 
            { id: 5, productName: "Generic Box B", db: 3, wb: 1, hb: 2, allowed_orientations: [] },
        ];
        sampleBoxes.forEach(addBoxEntry);
        // boxIdCounter is implicitly updated by addBoxEntry if !boxData.id
        // Call updateBoxIds to ensure all IDs (data-internal-id and checkbox/label for attributes) are sequential and correct
        updateBoxIds(); 
    }

    function handleTruckSelection() {
        const selectedIndex = truckSelector.value;
        if (selectedIndex === "") { // "-- Manually Enter Dimensions --" selected
            // Optionally clear the fields or leave them as is for manual input
            // containerDcInput.value = '';
            // containerWcInput.value = '';
            // containerHcInput.value = '';
            return;
        }

        const selectedTruck = allTrucks[parseInt(selectedIndex)];
        if (selectedTruck) {
            containerDcInput.value = parseFloat(selectedTruck["Longueur (m)"]).toFixed(3);
            containerWcInput.value = parseFloat(selectedTruck["Largeur (m)"]).toFixed(3);
            containerHcInput.value = parseFloat(selectedTruck["Hauteur (m)"]).toFixed(3);
        }
    }

    // Initial setup
    await Promise.all([loadProducts(), loadTrucks()]); // Load products and trucks in parallel
    setupApiDocs(); // Setup API documentation panel
    loadSampleBoxes();    // Then load sample boxes

    addBoxBtn.addEventListener('click', () => {
        addBoxEntry();
        updateBoxIds(); // Ensure new box gets correct sequential ID and updates others if needed
    });
    loadSampleBoxesBtn.addEventListener('click', loadSampleBoxes);


    solveBtn.addEventListener('click', async function() {
        loadingIndicator.style.display = 'block';
        visualizationsArea.innerHTML = '';
        metricsSummaryDiv.innerHTML = '';
        layerMetricsDetailsDiv.innerHTML = '';
        errorMessageDiv.style.display = 'none';
        errorMessageDiv.textContent = '';

        const containerData = {
            dc: containerDcInput.value,
            wc: containerWcInput.value,
            hc: containerHcInput.value
        };

        const boxesData = [];
        let inputValid = true;
        document.querySelectorAll('.box-entry').forEach(entry => {
            const productNameInput = entry.querySelector('.box-product-name');
            const id = productNameInput.dataset.internalId; // Use the internal ID
            const db = entry.querySelector('.box-db').value;
            const wb = entry.querySelector('.box-wb').value;
            const hb = entry.querySelector('.box-hb').value;

            const allowed_orientations = [];
            entry.querySelectorAll('.box-orientation:checked').forEach(checkbox => {
                allowed_orientations.push(parseInt(checkbox.value));
            });

            if (!id || !db || !wb || !hb || !productNameInput.value) { 
                inputValid = false;
            }
            boxesData.push({ id, db, wb, hb, allowed_orientations });
        });

        if (!inputValid || !containerData.dc || !containerData.wc || !containerData.hc) {
            errorMessageDiv.textContent = 'Error: Product name and all box dimensions (D, W, H) and container dimensions must be filled.';
            errorMessageDiv.style.display = 'block';
            loadingIndicator.style.display = 'none';
            return;
        }

        const gaSettings = {
            npop: parseInt(document.getElementById('ga-npop').value) || 50,
            nrep: parseInt(document.getElementById('ga-nrep').value) || 10,
            nmerge: parseInt(document.getElementById('ga-nmerge').value) || 10,
            max_generations: parseInt(document.getElementById('ga-max-gens').value) || 50,
            max_time_limit: parseInt(document.getElementById('ga-max-time').value) || 30
        };

        try {
            const response = await fetch('/solve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ container: containerData, boxes: boxesData, ga_settings: gaSettings }),
            });

            loadingIndicator.style.display = 'none';
            const result = await response.json();

            if (!response.ok || result.error) {
                let errorText = `Error: ${result.error || response.statusText}`;
                if (result.trace) { 
                    errorText += `\n\nServer Traceback (for debugging):\n${result.trace}`;
                }
                errorMessageDiv.textContent = errorText;
                errorMessageDiv.style.display = 'block';
                console.error('Error from server:', result);
                return;
            }
            
            displayMetrics(result.metrics);
            if (result.visualization_data && result.visualization_data.layers) {
                displayVisualizations(result.visualization_data);
            } else {
                 visualizationsArea.innerHTML = "<p>No visualization data returned or no layers in the plan.</p>";
            }

        } catch (error) {
            loadingIndicator.style.display = 'none';
            errorMessageDiv.textContent = 'Failed to fetch results from server: ' + error.message;
            errorMessageDiv.style.display = 'block';
            console.error('Fetch error:', error);
        }
    });

    function displayMetrics(metrics) {
        if (!metrics) {
            metricsSummaryDiv.innerHTML = "<p>No metrics data available.</p>";
            return;
        }
        metricsSummaryDiv.innerHTML = `
            <h3>Overall Performance</h3>
            <p><strong>Total Layers:</strong> ${metrics.total_layers}</p>
            <p><strong>Unstowed Box IDs:</strong> ${metrics.unstowed_box_ids && metrics.unstowed_box_ids.length > 0 ? metrics.unstowed_box_ids.join(', ') : 'None'}</p>
            <p><strong>Remaining Container Depth (xcfree):</strong> ${parseFloat(metrics.remaining_container_depth).toFixed(2)}</p>
            <p><strong>Total Packed Volume:</strong> ${parseFloat(metrics.total_packed_volume).toFixed(2)} / ${parseFloat(metrics.container_volume).toFixed(2)}</p>
            <p><strong>Overall Volume Utilization:</strong> ${metrics.overall_volume_utilization}</p>
        `;

        if (metrics.layer_details && metrics.layer_details.length > 0) {
            let detailsHtml = '<h3>Layer Details</h3>';
            metrics.layer_details.forEach(layer => {
                detailsHtml += `
                    <div class="layer-metrics-details-item">
                        <h4>Layer ${layer.layer_number}</h4>
                        <p><strong>Depth (d):</strong> ${parseFloat(layer.depth).toFixed(2)}</p>
                        <p><strong>Volume Utilization (vutil):</strong> ${layer.volume_utilization}</p>
                        <p><strong>Number of Boxes:</strong> ${layer.num_boxes}</p>
                        <p><strong>Box IDs:</strong> ${layer.box_ids.join(', ')}</p>
                    </div>
                `;
            });
            layerMetricsDetailsDiv.innerHTML = detailsHtml;
        } else {
            layerMetricsDetailsDiv.innerHTML = "<p>No detailed layer metrics available.</p>";
        }
    }

    function displayVisualizations(vizData) {
        visualizationsArea.innerHTML = ''; 
        const containerDims = vizData.container_dims;

        if (!vizData.layers || vizData.layers.length === 0) {
            visualizationsArea.innerHTML = "<p>No layers to visualize in the solution.</p>";
            return;
        }

        vizData.layers.forEach((layer, index) => {
            const layerDivId = `layer-viz-${index}`;
            const layerContainer = document.createElement('div');
            layerContainer.classList.add('layer-visualization-container');
            
            const title = document.createElement('h3');
            const startX = typeof layer.layer_start_x_ref === 'number' ? parseFloat(layer.layer_start_x_ref).toFixed(2) : layer.layer_start_x_ref;
            title.textContent = `Layer ${layer.layer_number} (Layer Depth Dimension: ${parseFloat(layer.layer_depth_dim).toFixed(2)}, Approx. Starts at X: ${startX})`;
            layerContainer.appendChild(title);

            const vizDiv = document.createElement('div');
            vizDiv.id = layerDivId;
            vizDiv.classList.add('plotly-graph-div'); 
            layerContainer.appendChild(vizDiv);
            visualizationsArea.appendChild(layerContainer);

            if (!layer.boxes || layer.boxes.length === 0) {
                vizDiv.innerHTML = "<p>This layer is empty.</p>";
                return;
            }

            const traces = [];
            layer.boxes.forEach(box => {
                const xCoords = [box.ox, box.ox + box.dx, box.ox + box.dx, box.ox, box.ox, box.ox + box.dx, box.ox + box.dx, box.ox];
                const yCoords = [box.oy, box.oy, box.oy + box.dy, box.oy + box.dy, box.oy, box.oy, box.oy + box.dy, box.oy + box.dy];
                const zCoords = [box.oz, box.oz, box.oz, box.oz, box.oz + box.dz, box.oz + box.dz, box.oz + box.dz, box.oz + box.dz];

                traces.push({
                    type: 'mesh3d',
                    x: xCoords, y: yCoords, z: zCoords,
                    i: [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j: [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k: [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    opacity: 0.8,
                    color: box.color, 
                    flatshading: true,
                    hoverinfo: 'text', 
                    text: `Box ID: ${box.id}<br>Pos: (${box.ox.toFixed(2)}, ${box.oy.toFixed(2)}, ${box.oz.toFixed(2)})<br>Dims: (${box.dx.toFixed(2)}, ${box.dy.toFixed(2)}, ${box.dz.toFixed(2)})`
                });
            });
            
            const layout = {
                margin: { l: 10, r: 10, b: 10, t: 40 }, 
                scene: {
                    xaxis: { title: 'X (Depth)', range: [0, containerDims.dc], autorange: false },
                    yaxis: { title: 'Y (Width)', range: [0, containerDims.wc], autorange: false },
                    zaxis: { title: 'Z (Height)', range: [0, containerDims.hc], autorange: false },
                    aspectmode: 'cube',
                    camera: {
                        eye: {x: 1.5, y: 1.5, z: 1.5} 
                    }
                },
                 autosize: true 
            };

            Plotly.newPlot(layerDivId, traces, layout).then(function(gd) {
                Plotly.Plots.resize(gd);
            });
        });
        
        window.addEventListener('resize', () => {
            document.querySelectorAll('.plotly-graph-div').forEach(div => {
                if (div.id && Plotly.Plots.getPlot(div.id)) { 
                     Plotly.Plots.resize(div.id);
                }
            });
        });
    }
    function setupApiDocs() {
        if (apiDocsOpenBtn && apiDocsPanel && apiDocsCloseBtn && apiDocsContentArea) {
            apiDocsOpenBtn.addEventListener('click', () => {
                apiDocsPanel.classList.add('open');
            });

            apiDocsCloseBtn.addEventListener('click', () => {
                apiDocsPanel.classList.remove('open');
            });

            // Optional: Close panel if user clicks outside of it
            document.addEventListener('click', (event) => {
                if (apiDocsPanel.classList.contains('open') && 
                    !apiDocsPanel.contains(event.target) && 
                    event.target !== apiDocsOpenBtn) {
                    apiDocsPanel.classList.remove('open');
                }
            });

            // Populate API Documentation Content
            apiDocsContentArea.innerHTML = `
                <p>The Bin Packing Optimizer can be integrated via its API endpoint. This allows external systems to submit packing problems and receive optimized solutions.</p>
                
                <h3>Endpoint: <code>/solve</code></h3>
                <p><strong>Method:</strong> <code>POST</code></p>
                <p><strong>Content-Type:</strong> <code>application/json</code></p>

                <h4>Request Body Structure:</h4>
                <p>The request body should be a JSON object with the following structure:</p>
                <pre><code>{
  "container": {
    "dc": 10.0,  // Container Depth (meters)
    "wc": 5.0,   // Container Width (meters)
    "hc": 8.0    // Container Height (meters)
  },
  "boxes": [
    {
      "id": 1,         // Unique Box ID (integer)
      "db": 2.0,       // Box Depth (meters)
      "wb": 3.0,       // Box Width (meters)
      "hb": 1.0,       // Box Height (meters)
      "allowed_orientations": [1, 2, 3, 4, 5, 6] // Optional: list of allowed rotation IDs (1-6)
    },
    {
      "id": 2,
      "db": 1.0,
      "wb": 1.0,
      "hb": 4.0
      // allowed_orientations defaults to all if not provided
    }
    // ... more boxes
  ],
  "ga_settings": { // Optional: Genetic Algorithm settings
    "npop": 50,
    "nrep": 10,
    "nmerge": 10,
    "max_generations": 50,
    "max_time_limit": 30 // seconds
  }
}</code></pre>

                <h4>Success Response (200 OK):</h4>
                <p>The response will be a JSON object containing metrics and visualization data. Example structure:</p>
                <pre><code>{
  "metrics": { /* ... detailed metrics ... */ },
  "visualization_data": { /* ... data for plotting layers ... */ },
  "status": "success"
}</code></pre>

                <h4>Error Response (e.g., 400 Bad Request, 500 Internal Server Error):</h4>
                <p>Error responses will also be in JSON format:</p>
                <pre><code>{
  "error": "Descriptive error message",
  "trace": "Optional server traceback for debugging" // If applicable
}</code></pre>
                <p>Ensure your integration handles these API responses appropriately.</p>
            `;
        }
    }
});

