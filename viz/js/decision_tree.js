/*
 * Project: CIS 678 Project 3 - Decision Tree
 * Authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
 * Purpose: Interactive Decision Tree using d3.js
 */

/*
 * References:
 *
 * http://bl.ocks.org/d3noob/8375092
 * http://bl.ocks.org/d3noob/8326869
 */ 

var data; // a global

d3.json(datasetFile, function(error, json) {
    if (error) return console.warn(error);
    data = [json]; // store object inside array
    generate();
});

var tree;
var diagonal;
var svg;
var root;
var i;
var duration;

// color styling
var nodeFillChildren = "#CE0000";
var nodeFillNoChildren = "#DEE7EF";

function generate() {
// ************** Generate the tree diagram	 *****************
    var margin = {top: 50, right: 50, bottom: 50, left: 50},
        width = 800 - margin.right - margin.left,
        height = 800 - margin.top - margin.bottom;

    i = 0;
    duration = 500; // milliseconds for animation

    tree = d3.layout.tree()
        .size([height, width]);

    // http://stackoverflow.com/questions/15007877/how-to-use-the-d3-diagonal-function-to-draw-curved-lines
    diagonal = d3.svg.diagonal()
        .projection(function (d) {
            return [d.x, d.y];  // top-down
        });

    svg = d3.select("#decision_tree").append("svg")
        .attr("width", width + margin.right + margin.left)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // set the root
    root = data[0];
    root.x0 = width / 2;
    root.y0 = 0;

    update(root);

    // http://stackoverflow.com/questions/22448032/d3-what-is-the-self-as-in-d3-selectself-frameelement-styleheight-height/22449389#22449389
    d3.select(self.frameElement).style("height", "500px");
}

function update(source) {

    // Compute the new tree layout.
    var nodes = tree.nodes(root).reverse(),
        links = tree.links(nodes);

    // Normalize for fixed-depth.
    nodes.forEach(function(d) { d.y = d.depth * 80; }); // depth of each level of tree

    // Update the nodes…
    var node = svg.selectAll("g.node")
        .data(nodes, function(d) { return d.id || (d.id = ++i); });

    // Enter any new nodes at the parent's previous position.
    var nodeEnter = node.enter().append("g")
        .attr("class", "node")
        .attr("transform", function(d) { return "translate(" + source.x0 + "," + source.y0 + ")"; })
        .on("click", click);

    nodeEnter.append("circle")
        .attr("r", 1e-6)
        .style("fill", function(d) { return d._children ? nodeFillChildren : nodeFillNoChildren; });

    nodeEnter.append("text")
        .attr("y", function(d) { return d.children || d._children ? -18 : 18; })
        .attr("dy", ".35em")
        .attr("text-anchor", function(d) { return d.children || d._children ? "middle" : "middle"; })
        .text(function(d) { return d.name; })
        .style("fill-opacity", 1)
        .style("font-weight", "bold");

    // Transition nodes to their new position.
    var nodeUpdate = node.transition()
        .duration(duration)
        .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });

    nodeUpdate.select("circle")
        .attr("r", 10)
        .style("fill", function(d) { return d._children ? nodeFillChildren : nodeFillNoChildren; });

    nodeUpdate.select("text")
        .style("fill-opacity", 1);

    // Transition exiting nodes to the parent's new position.
    var nodeExit = node.exit().transition()
        .duration(duration)
        .attr("transform", function(d) { return "translate(" + source.x + "," + source.y + ")"; })
        .remove();

    nodeExit.select("circle")
        .attr("r", 1e-6);

    nodeExit.select("text")
        .style("fill-opacity", 1e-6);

    // Update the links…
    var link = svg.selectAll("path.link")
        .data(links, function(d) { return d.target.id; });

    // Enter any new links at the parent's previous position.
    link.enter().insert("path", "g")
        .attr("class", "link")
        .attr("d", function(d) {
            var o = {x: source.x0, y: source.y0};
            return diagonal({source: o, target: o});
        });

    // Transition links to their new position.
    link.transition()
        .duration(duration)
        .attr("d", diagonal);

    // Transition exiting nodes to the parent's new position.
    link.exit().transition()
        .duration(duration)
        .attr("d", function(d) {
            var o = {x: source.x, y: source.y};
            return diagonal({source: o, target: o});
        })
        .remove();

    // Stash the old positions for transition.
    nodes.forEach(function(d) {
        d.x0 = d.x;
        d.y0 = d.y;
    });
}

// Toggle children on click.
function click(d) {
    if (d.children) {
        d._children = d.children;
        d.children = null;
    } else {
        d.children = d._children;
        d._children = null;
    }
    update(d);
}
