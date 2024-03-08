function sendQuery(){
    query = document.getElementById('textbox').value;
    window.location.assign('/result'+ (query ? '/'+query : ''));
};

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
