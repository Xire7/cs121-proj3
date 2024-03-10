function sendQuery(){
    query = document.getElementById('textbox').value;
    console.log("sent");
    window.location.assign('/result'+ (query ? '/'+query : ''));
};
/*function sendQuery() {
    var query = document.getElementById('textbox').value;
    var url = '/result' + (query ? '?query=' + encodeURIComponent(query) : '');

    // Making an AJAX request to your Python backend
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // Assuming your backend returns JSON data
            var results = JSON.parse(xhr.responseText);
            // Getting the first 20 indices of results
            var first20Results = results.slice(0, 20);
            // Do something with first 20 results, like display them
            console.log(first20Results);
        }
    };
    xhr.send();
};*/
/*function sendQuery() {
    const query = document.getElementById('textbox').value;
  
    fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
      // Process the received search results (data) in the frontend
      console.log(data); // Example: display results in the UI
    })
    .catch(error => {
      console.error('Error fetching results:', error);
    });
  }*/

// function incrementCounter(){
//     let index = parseInt(document.getElementById('index').innerText);
//     // Increment the number attribute
//     index += 1;
//     // Update the content of the HTML element to display the incremented number
//     document.getElementById('index').innerText = index;
//     window.location.reload();
// };

let currentIndex = 0; // Initialize the current index to 0

function incrementCounter() {
    let length = parseInt(document.getElementById('length').innerText);
    console.log(length);
    // Increment the current index if possible
    if(currentIndex < length - 1){
        currentIndex++;
    }

    // Update the content of the HTML element to display the incremented index
    document.getElementById('index').innerText = currentIndex + 1;
    // Hide all lists
    document.querySelectorAll('.url-list').forEach(function(list) {
        list.style.display = 'none';
    });
    // Show the list corresponding to the current index
    document.getElementById('list-' + currentIndex).style.display = 'block';
}

function decrementCounter() {
    let length = parseInt(document.getElementById('length').innerText);
    console.log(length);
    // Decrement the current index if possible
    if(currentIndex > 0){
        currentIndex--;
    }
    console.log(currentIndex);
    // Update the content of the HTML element to display the incremented index
    document.getElementById('index').innerText = currentIndex + 1;
    // Hide all lists
    document.querySelectorAll('.url-list').forEach(function(list) {
        list.style.display = 'none';
    });
    // Show the list corresponding to the current index
    document.getElementById('list-' + currentIndex).style.display = 'block';
}
