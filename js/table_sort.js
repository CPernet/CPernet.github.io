document.addEventListener("DOMContentLoaded", function () {
    initializeRowNumbers();
});

var sortColumn = -1; // Initial sort column
var sortDirection = 1; // Initial sort direction is ascending

function sortTable(columnIndex) {
    var table, rows, switching, i, shouldSwitch;
    table = document.getElementById("sortableTable");
    switching = true;

    if (sortColumn === columnIndex) {
        // If clicking on the same column, toggle the sort direction
        sortDirection *= -1;
    } else {
        // If clicking on a different column, set the new column and reset the sort direction
        sortColumn = columnIndex;
        sortDirection = 1;
    }

    while (switching) {
        switching = false;
        rows = table.rows;

        for (i = 1; i < rows.length - 1; i++) {
            shouldSwitch = false;

            var x = rows[i].getElementsByTagName("td")[sortColumn];
            var y = rows[i + 1].getElementsByTagName("td")[sortColumn];

            var xValue = x.innerHTML.toLowerCase();
            var yValue = y.innerHTML.toLowerCase();

            if ((sortDirection === 1 && xValue > yValue) || (sortDirection === -1 && xValue < yValue)) {
                shouldSwitch = true;
                break;
            }
        }

        if (shouldSwitch) {
            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
            switching = true;
        }
    }

    // Add row numbers
    initializeRowNumbers();
}

function initializeRowNumbers() {
    var table = document.getElementById("sortableTable");
    var rows = table.rows;

    // Add row numbers
    for (var i = 1; i < rows.length; i++) {
        rows[i].getElementsByTagName("td")[0].innerHTML = i;
    }
}
