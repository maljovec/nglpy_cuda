
//In case the CSS does not define the properties for styling our graph, then
// we should use some of our own
var defaultActiveColor = "steelblue";
var defaultHighlightColor = "orange";
var defaultInactiveColor = "white";
var defaultActiveOpacity = 0.5;
var defaultInactiveOpacity = 0.25;

/**
 * Construct a generic grid with some background gridlines
 * @param {*} canvas - svg element to write on
 * @param {*} width - Desired width of your plot
 * @param {*} height - Desired width of your plot
 * @param {*} xCells - Number of cells laid out horizontally
 * @param {*} yCells - Number of cells laid out vertically
 */
function constructGrid(canvas, width, height, xCells=5, yCells = 5) {
    var style = window.getComputedStyle(document.body);
    var borderColor = style.getPropertyValue('--active-color', defaultActiveColor);

    var xCellSize = 1/xCells;
    var yCellSize = 1/yCells;
    var grid = new Array(xCells * yCells);

    canvas.attr("width", width)
        .attr("height", height)
        .style("display", "block")
        .style("margin", "auto")
        .style("border", "1px solid " + borderColor)
        .append("path")
            .attr("class", "grid")
            .attr("d", d3.range(xCellSize*width, width, xCellSize*width)
                            .map(function(x) { return "M" + Math.round(x) + ",0V" + height; })
                            .join("")
                    + d3.range(yCellSize*height, height, yCellSize*height)
                        .map(function(y) { return "M0," + Math.round(y) + "H" + width; })
                        .join(""));
}

function cmp(a,b) {
    return a[0] - b[0];
}

// A convenience structure for filling in polygons that are specified as
// paths (This may not be the best way of doing this)
var area = d3.line()
            .x(function(d){ return d.x; })
            .y(function(d){ return d.y; });

/**
 * Callback function for highlighting a given element's neighbors
 * TODO: Remove base name as the canvas should now only have one set of "points"
 * for example
 * @param {*} canvas
 * @param {*} base
 * @param {*} i
 * @param {*} highlighted
 */
function toggleNeighbor(canvas, base, i, highlighted) {
    var style = window.getComputedStyle(document.body);
    var colorOn = style.getPropertyValue('--highlight-color', defaultHighlightColor);
    var colorOff = style.getPropertyValue('--active-color', defaultActiveColor);
    var opacityOn = style.getPropertyValue('--active-opacity', defaultActiveOpacity);
    var opacityOff = style.getPropertyValue('--inactive-opacity', defaultInactiveOpacity);
    var idx = i.toString();

    var pointColor, fillColor, fillOpacity, strokeColor, strokeOpacity;

    if (highlighted)
    {
        pointColor = colorOn;
        fillColor = colorOn;
        fillOpacity = opacityOn;
        strokeColor = colorOn;
        strokeOpacity = '1';
    }
    else
    {
        pointColor = colorOff;
        fillColor = '#fff';
        fillOpacity = '0';
        strokeColor = colorOff;
        strokeOpacity = opacityOff;
    }

    canvas.selectAll("#"+base+"_Point_" + idx)
        .transition()
            .style("fill", pointColor);

    canvas.selectAll("#"+base+"_Circle_" + idx)
        .transition()
            .style("fill", fillColor)
            .style("fill-opacity", fillOpacity)
            .style("stroke", strokeColor)
            .style("stroke-opacity", strokeOpacity);
}

/**
 * TODO: Remove dependency on global points array (get it from the canvas)
 * @param {*} canvas
 * @param {*} item
 * @param {*} highlighted
 */
function knnToggleItem(canvas, item, highlighted) {
    var style = window.getComputedStyle(document.body);
    var colorOn = style.getPropertyValue('--highlight-color', defaultHighlightColor);
    var colorOff = style.getPropertyValue('--active-color', defaultActiveColor);
    var tokens = item.split('_');

    var pointColor, fillColor, fillOpacity, strokeColor, strokeOpacity;
    //Point highlighted
    if (tokens.length == 3) {
        if (highlighted) {
            pointColor = colorOn;
            fillColor = colorOn;
            fillOpacity = '0.25';
            strokeColor = colorOn;
            strokeOpacity = '1';
        } else {
            pointColor = colorOff;
            fillColor = '#fff';
            fillOpacity = '0';
            strokeColor = colorOff;
            strokeOpacity = '0.15';
        }
        canvas.selectAll("#"+item)
            .transition()
            .style("fill", pointColor);

        var i = parseInt(tokens[2]);
        var x1 = points[2*i];
        var y1 = points[2*i+1];
        var distances = [];
        for (var j = 0; j < numSamples; j++)
        {
            if(i == j)
            {
                continue;
            }
            var x2 = points[2*j];
            var y2 = points[2*j+1];
            var distance = lpNorm(x1,y1,x2,y2);
            distances.push([distance,j]);
        }
        distances.sort(cmp);
        for (var j = 0; j < k; j++ )
        {
            idx = distances[j][1];
            if (i < idx)
            {
                idx1 = i;
                idx2 = idx;
            }
            else
            {
                idx1 = idx;
                idx2 = i;
            }
            canvas.selectAll("#" + tokens[0] + '_Edge_' + idx1.toString() + '_' + idx2.toString())
            .transition()
                .style("stroke", strokeColor)
                .style("fill", fillColor);
            toggleNeighbor(canvas, tokens[0], idx, highlighted);
            if (j == k-1) {
            canvas.selectAll("#" + tokens[0] + '_Circle_' + i.toString() + '_' + idx.toString())
            .transition()
                .style("fill", fillColor)
                .style("fill-opacity", fillOpacity)
                .style("stroke", strokeColor)
                .style("stroke-opacity", strokeOpacity);
            }
        }
    }
    else
    {
        if (highlighted)
        {
            strokeColor = colorOn;
            fillColor = colorOn;
        }
        else
        {
            strokeColor = colorOff;
            fillColor = colorOff;
        }

        canvas.selectAll("#" + item)
            .transition()
            .style("stroke", strokeColor)
            .style("fill", fillColor);

        for (idx = 2; idx < tokens.length; idx++)
        {
            toggleNeighbor(canvas, tokens[0],parseInt(tokens[idx]), highlighted);
        }
    }
}

function ergToggleItem(canvas, item, highlighted) {
    var colorOn = style.getPropertyValue('--highlight-color', defaultHighlightColor);
    var colorOff = style.getPropertyValue('--active-color', defaultActiveColor);
    var opacityOn = style.getPropertyValue('--active-opacity', defaultActiveOpacity);
    var opacityOff = style.getPropertyValue('--inactive-opacity', defaultInactiveOpacity);

    var tokens = item.split('_');

    var pointColor, fillColor, fillOpacity, strokeColor, strokeOpacity;

    //Point highlighted
    if (tokens.length == 3) {
        if (highlighted)
        {
            pointColor = colorOn;
            fillColor = colorOn;
            fillOpacity = '0.25';
            strokeColor = colorOn;
            strokeOpacity = '1';
        }
        else
        {
            pointColor = colorOff;
            fillColor = '#fff';
            fillOpacity = '0';
            strokeColor = colorOff;
            strokeOpacity = '0.15';
        }

        canvas.selectAll("#"+item)
            .transition()
            .style("fill", pointColor);

        canvas.selectAll("#"+tokens[0]+"_Circle_" + tokens[2])
            .transition()
            .style("fill", fillColor)
            .style("fill-opacity", fillOpacity)
            .style("stroke", strokeColor)
            .style("stroke-opacity", strokeOpacity);
    }
    else
    {
        if (highlighted)
        {
            strokeColor = colorOn;
            fillColor = colorOn;
            fillOpacity = '0.25';
            strokeOpacity = '1';
        }
        else
        {
            strokeColor = colorOff;
            fillColor = colorOff;
            fillOpacity = '0';
            strokeOpacity = '0.15';
        }

        canvas.selectAll("#" + item)
            .transition()
            .style("stroke", strokeColor)
            .style("fill", fillColor);

        canvas.selectAll("#"+tokens[0]+"_Circle_" + tokens[2]+'_'+tokens[3])
            .transition()
            .style("fill", fillColor)
            .style("fill-opacity", fillOpacity)
            .style("stroke", strokeColor)
            .style("stroke-opacity", strokeOpacity);

        canvas.selectAll("#"+tokens[0]+"_Circle_" + tokens[3]+'_'+tokens[2])
            .transition()
            .style("fill", fillColor)
            .style("fill-opacity", fillOpacity)
            .style("stroke", strokeColor)
            .style("stroke-opacity", strokeOpacity);

        if (highlighted) {
            strokeColor = colorOn;
            fillColor = colorOn;
        }
        else
        {
            strokeColor = '#999';
            fillColor = '#999';
        }

        canvas.selectAll("#"+tokens[0]+"_GuidePoint_" + tokens[2]+'_'+tokens[3])
            .transition()
            .style("stroke", strokeColor)
            .style("fill", fillColor);

        canvas.selectAll("#"+tokens[0]+"_GuidePoint_" + tokens[3]+'_'+tokens[2])
            .transition()
            .style("stroke", strokeColor)
            .style("fill", fillColor);

        for (idx = 2; idx < tokens.length; idx++)
        {
            toggleNeighbor(canvas, tokens[0], parseInt(tokens[idx]), highlighted);
        }
    }
}

function knnMouseOut(element, canvas) {
    knnToggleItem(canvas, element.id.toString(), false);
}
function knnMouseOver(element, canvas) {
    knnToggleItem(canvas, element.id.toString(), true);
}

function ergMouseOut(element, canvas) {
    ergToggleItem(canvas, element.id.toString(), false);
}
function ergMouseOver(element, canvas) {
    ergToggleItem(canvas, element.id.toString(), true);
}

function constructKNN(canvas, points, k, pNorm) {
    canvas.selectAll('.circle').remove();
    canvas.selectAll('.edge').remove();
    canvas.selectAll('.point').remove();

    var knnCircles = canvas.append("g").attr("class", "circle");
    var knnEdges = canvas.append("g").attr("class", "edge");
    var knnPoints = canvas.append("g").attr("class", "point");

    //Edges first
    //Brute force approach, this problem is small enough.
    for (var i = 0; i < numSamples; i++)
    {
        var x1 = points[2*i];
        var y1 = points[2*i+1];
        var distances = [];
        for (var j = 0; j < numSamples; j++)
        {
            if (i == j)
            {
                continue;
            }
            var x2 = points[2*j];
            var y2 = points[2*j+1];
            var distance = lpNorm(x1,y1,x2,y2);
            distances.push([distance,j]);
        }
        distances.sort(cmp);
        for (var j = 0; j < k; j++ )
        {
            var idx = distances[j][1];
            if (i < idx)
            {
                idx1 = i;
                idx2 = idx;
            }
            else
            {
                idx1 = idx;
                idx2 = i;
            }

            var x2 = points[2*idx];
            var y2 = points[2*idx+1];

            knnEdges.append("line")
            .attr("class","edge")
            .attr("x1",x1)
            .attr("y1",y1)
            .attr("x2",x2)
            .attr("y2",y2)
            .attr("id","knn_Edge_"+idx1.toString()+"_"+idx2.toString())
            .on("mouseover", function(){ return knnMouseOver(this, canvas); } )
            .on("mouseout", function(){ return knnMouseOut(this, canvas); } );

            if (j == k-1)
            {
                var radius = distances[j][0];
                var maxNeighborShape = pNormShape(pNorm, x1, y1, radius);

                knnCircles.append("path")
                        .datum(maxNeighborShape)
                        .attr("d", area)
                        .attr('class','polygon')
                        .attr("id","knn_Circle_"+i.toString()+'_'+idx.toString());
            }
        }
    }

    //Points second
    for (var i = 0; i < numSamples; i++)
    {
        var x = points[2*i];
        var y = points[2*i+1];

        knnPoints.append("circle")
            .attr("class", "point")
            .attr("r", 1e-6)
            .attr("cx", x)
            .attr("cy", y)
            .attr("id","knn_Point_"+i.toString())
            .on("mouseover", function(){ return knnMouseOver(this, canvas); } )
            .on("mouseout", function(){ return knnMouseOut(this, canvas); } )
            .attr("r", 5);
    }
}

function constructERG(canvas, points, beta, pNorm) {
    canvas.selectAll('.circle').remove();
    canvas.selectAll('.face').remove();
    canvas.selectAll('.edge').remove();
    canvas.selectAll('.point').remove();
    canvas.selectAll('.guidePoint').remove();

    var ergCircles = canvas.append("g").attr("class", "circle");
    var ergEdges = canvas.append("g").attr("class", "edge");
    var ergPoints = canvas.append("g").attr("class", "point");
    var ergGuidePoints = canvas.append("g").attr("class", "guidePoint");

    numSamples = points.length/2;
    //Edges first
    //Brute force approach, this problem is small enough.
    for (var i = 0; i < numSamples; i++)
    {
        var x1 = points[2*i];
        var y1 = points[2*i+1];
        for (var j = i+1; j < numSamples; j++)
        {
            var x2 = points[2*j];
            var y2 = points[2*j+1];
            var radius = 0.5*lpNorm(x1,y1,x2,y2,2);
            var xC = 0.5*(points[2*j]+points[2*i]);
            var yC = 0.5*(points[2*j+1]+points[2*i+1]);
            var valid = true;
            for (var ii = 0; ii < numSamples; ii++)
            {
                if (i == ii || j == ii)
                {
                    continue;
                }
                var x3 = points[2*ii];
                var y3 = points[2*ii+1];
                if (beta < 1)
                {
                    var r = radius / beta;
                    var delta = Math.sqrt(Math.pow(r,2) - (Math.pow(radius,2)));

                    var PR = [x3-x1,y3-y1];
                    var PQ = [x2-x1,y2-y1];
                    var qpNorm = lpNorm(x1,y1,x2,y2,2);
                    var t = dot(PR,PQ) / dot(PQ,PQ);
                    var proj = [t*x2+(1-t)*x1, t*y2+(1-t)*y1];

                    var dproj = lpNorm(x3, y3, proj[0], proj[1], 2);
                    var dprojmidsqr = Math.pow(lpNorm(proj[0], proj[1], 0.5*(x1+x2), 0.5*(y1+y2), 2), 2);

                    var d2 = dprojmidsqr + (dproj + delta)*(dproj + delta);
                    var d = Math.sqrt(d2);
                    if (d < r)
                    {
                        valid = false;
                        break;
                    }
                }
                else
                {
                    var c1 = [(1-beta/2)*x1+(beta/2)*x2,(1-beta/2)*y1+(beta/2)*y2];
                    var c2 = [(beta/2)*x1+(1-beta/2)*x2,(beta/2)*y1+(1-beta/2)*y2];
                    var r2 = Math.pow(radius,2)*Math.pow(beta,2);
                    var d1 = Math.pow(x3-c1[0],2)+Math.pow(y3-c1[1],2);
                    var d2 = Math.pow(x3-c2[0],2)+Math.pow(y3-c2[1],2);
                    if (Math.max(d1,d2) < r2)
                    {
                        valid = false;
                        break;
                    }
                }
            }
            if (valid)
            {
                var base = 'erg_';
                var iStr = i.toString();
                var jStr = j.toString();

                //IDs for each of the drawn components
                var intersectID = base+"Circle_"+iStr+'_'+jStr;

                var guidePtID1 = base+"GuidePoint_"+iStr+"_"+jStr;
                var guidePtID2 = base+"GuidePoint_"+jStr+"_"+iStr;

                var guideID1 = base+"Guide_"+iStr+'_'+jStr;
                var guideID2 = base+"Guide_"+jStr+'_'+iStr;

                var edgeID = base+"Edge_"+iStr+"_"+jStr;

                ergEdges.append("line")
                    .attr("class","edge")
                    .attr("x1",x1)
                    .attr("y1",y1)
                    .attr("x2",x2)
                    .attr("y2",y2)
                    .attr("id",edgeID)
                    .on("mouseover", function(){ return ergMouseOver(this, canvas); } )
                    .on("mouseout", function(){ return ergMouseOut(this, canvas); } );

                var theta = Math.atan2(y2-y1,x2-x1);
                // var p = [{x: x1, y: y1}];
                var q = {x: x2, y: y2};
                var c = {x: xC, y: yC};

                var cR = {x: c.x, y: c.y};
                cR = translate(cR, -x1, -y1);
                cR = rotate(cR, -theta);
                cR = translate(cR, x1, y1);

                var qR = {x: q.x, y: q.y};
                qR = translate(qR, -x1, -y1);
                qR = rotate(qR, -theta);
                qR = translate(qR, x1, y1);

                var r;
                var xc1,yc1,xc2,yc2;

                if (beta < 1) {
                    r = lpNorm(x1, y1, cR.x, cR.y, pNorm) / beta;

                    var a = Math.pow(Math.pow(r,pNorm)-Math.pow(radius,pNorm),1./pNorm);
                    var b = lpNorm(x1, y1, qR.x, qR.y, pNorm);

                    xc1 = xC + a*(y2-y1)/b;
                    yc1 = yC + a*(x1-x2)/b;

                    xc2 = xC - a*(y2-y1)/b;
                    yc2 = yC - a*(x1-x2)/b;

                }
                else
                {
                    r = lpNorm(x1, y1, cR.x, cR.y, pNorm) * beta;
                    //r = lpNorm(x1, y1, xC, yC, pNorm) * beta;

                    xc1 = (1-0.5*beta)*x1 + 0.5*beta*x2;
                    yc1 = (1-0.5*beta)*y1 + 0.5*beta*y2;

                    xc2 = (1-0.5*beta)*x2 + 0.5*beta*x1;
                    yc2 = (1-0.5*beta)*y2 + 0.5*beta*y1;
                }

                var er1 = pNormShape(pNorm, 0, 0, r, false);
                er1.map(function(x) { return rotate(x,theta)});
                er1.map(function(x) { return translate(x,xc1,yc1)});

                var er2 = pNormShape(pNorm, 0, 0, r, false);
                er2.map(function(x) { return rotate(x,theta)});
                er2.map(function(x) { return translate(x,xc2,yc2)});

                if (debug)
                {
                    ergCircles.append("polygon")
                        .attr("points", pointsToString(er1))
                        .attr('class','face')
                        .attr("id", guideID1);
                    ergCircles.append("polygon")
                        .attr("points", pointsToString(er2))
                        .attr('class','face')
                        .attr("id",guideID2);
                }

                var er1_n_er2 = greinerHormann.intersection(er1, er2);

                if (er1_n_er2)
                {
                    er1_n_er2.forEach(function(d){
                                ergCircles.append("path")
                                            .datum(d)
                                            .attr("d", area)
                                            .attr("class", "polygon")
                                            .attr("id", intersectID)});
                }

                if (debug)
                {
                    ergGuidePoints.append("circle")
                        .attr("class", "guidePoint")
                        .attr("r", 5)
                        .attr("cx", xc1)
                        .attr("cy", yc1)
                        .attr("id",guidePtID1);
                    ergGuidePoints.append("circle")
                        .attr("class", "guidePoint")
                        .attr("r", 5)
                        .attr("cx", xc2)
                        .attr("cy", yc2)
                        .attr("id",guidePtID2);
                }
            }
        }
    }

    //Points second
    for (var i = 0; i < numSamples; i++)
    {
        var x = points[2*i];
        var y = points[2*i+1];

        ergPoints.append("circle")
            .attr("class", "point")
            .attr("r", 5)
            .attr("cx", x)
            .attr("cy", y)
            .attr("id","erg_Point_"+i.toString())
            .on("mouseover", function(){ return ergMouseOver(this, canvas); } )
            .on("mouseout", function(){ return ergMouseOut(this, canvas); } );
    }
}