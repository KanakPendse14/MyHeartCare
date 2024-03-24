// scripts.js

document.addEventListener('DOMContentLoaded', function() {
    const fetchDataBtn = document.getElementById('fetchDataBtn');
    const messageElement = document.getElementById('message');

    fetchDataBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/api');
            const data = await response.json();
            messageElement.textContent = data.message;
        } catch (error) {
            console.error('Error fetching data:', error);
            messageElement.textContent = 'Error fetching data. See console for details.';
        }
    });
});
