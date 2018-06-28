/**
 * A simple dot product for two one-dimensional arrays
 * @param {array} a - The first array
 * @param {array} b - The second array
 */
function dot(a, b) {
    var summand = 0;
    var lim = Math.min(a.length, b.length);
    for (var i = 0; i < lim; i++) {
        summand += a[i] * b[i];
    }
    return summand;
}

/**
 * A function for computing the Lp-norm of a shape
 * @param {*} x1
 * @param {*} y1
 * @param {*} x2
 * @param {*} y2
 * @param {*} p
 */
function lpNorm(x1, y1, x2, y2, p = pNorm) {
    if (p == Number.POSITIVE_INFINITY) {
        return Math.max(Math.abs(x1 - x2), Math.abs(y1 - y2));
    }
    else {
        return Math.pow(Math.pow(Math.abs(x1 - x2), p) + Math.pow(Math.abs(y1 - y2), p), 1 / p);
    }

}

/**
 * Constructs a list of points that outline the planar shape of the specified
 * Lp-norm
 * @param {float} p - The power parameter of the Minkowski norm
 * @param {float} xc - The horizontal center of the shape
 * @param {float} yc - The vertical center of the shape
 * @param {float} radius - The distance from the outer edge to the center point
 */
function pNormShape(p, xc = 0, yc = 0, radius = 1, closed = true) {
    var numSegments = 180;
    var pts = [];

    if (p == Number.POSITIVE_INFINITY) {
        pts.push({ x: xc + radius, y: yc + radius });
        pts.push({ x: xc - radius, y: yc + radius });
        pts.push({ x: xc - radius, y: yc - radius });
        pts.push({ x: xc + radius, y: yc - radius });
    }
    else {
        for (var i = 0; i < numSegments; i++) {
            var x = radius * (i / numSegments * 2.0 - 1.0);

            var radiusP = Math.pow(radius, p);
            var xP = Math.pow(Math.abs(x), p);

            var yPos = Math.pow(radiusP - xP, 1 / p);

            pts.unshift({ x: xc + x, y: yc + yPos });
            pts.push({ x: xc + x, y: yc - yPos });
        }
    }

    //Add the first point to the end to ensure a closed path
    if (closed) {
        pts.push({ x: pts[0].x, y: pts[0].y });
    }
    return pts;
}

/**
 * A convenience function that will take the arrays of points of the form
 * [{x: x0, y: y0}, ..., {x: xn, y: yn}] and turn them into a flattened string
 * of the form "x0,y0,...,xn,yn".
 * @param {array} A - Array to parse (should be made of Objects having x
 *                    and y as properties, all other fields will be ignored)
 */
function pointsToString(A) {
    var points = ""
    var sep = ''
    for (let a of A) {
        points += sep + a.x.toString() + ',' + a.y.toString();
        sep = ','
    }
    return points;
}

/**
 * Perform a 2D translation on a collection of 2D points
 * @param {array} A - Array of points to translate (should be made of Objects
 *                    having x and y as properties, all other fields will be
 *                    ignored)
 * @param {float} x - The amount to translate in the horizontal direction
 * @param {float} y - The amount to translate in the vertical direction
 */
function translate(A, x, y) {
    A.x = A.x + x;
    A.y = A.y + y;
    return A;
}

/**
 * Perform a 2D rotation on a collection of 2D points
 * @param {array} A - Array of points to rotate (should be made of Objects
 *                    having x and y as properties, all other fields will be
 *                    ignored)
 * @param {float} theta - The angle in radians to rotate the points about the
 *                        origin
 */
function rotate(A, theta) {
    var sT = Math.sin(theta);
    var cT = Math.cos(theta);
    var x = A.x;
    var y = A.y;
    A.x = cT * x - sT * y;
    A.y = sT * x + cT * y;
    return A;
}
////////////////////////////////////////////////////////////////////////////////
// WIP

function maxDistance(t, beta, p) {
    // Handle the symmetry of the problem by centering on the midpoint and
    // considering only half the problem space, note the scale of t has been
    // doubled, so whatever we compute should be halved to get it back into
    // the original space
    t = Math.abs(2 * t - 1);

    var xC = 0;
    var yC = 0;
    var r;

    if (t > 1) { return 0; }

    if (beta <= 1) {
        r = 1 / beta;
        yC = Math.pow(Math.pow(r, p) - 1, 1 / p);
    }
    else {
        r = beta;
        xC = 1 - beta;
    }
    ////// DEBUG
    // d3.selectAll('#debugPoint').remove();
    // d3.selectAll('#debugShape').remove();

    // d3.select('#templatePlot')
    //     .append("circle")
    //         .attr("class", "guidePoint")
    //         .attr("id", "debugPoint")
    //         .attr('cx', 100*xC+100)
    //         .attr('cy', 200-100*yC)
    //         .attr('r', 5);

    // var template = pNormShape(pNorm, 100+100*xC, 200+100*yC, 100*r);
    // d3.select('#templatePlot')
    //     .append("polygon")
    //         .attr("points", pointsToString(template))
    //         .attr("class", "guidePoint")
    //         .attr("id", "debugShape");

    // template = beta < 1 ? pNormShape(pNorm, 100+100*xC, 200-100*yC, 100*r)
    //                     : pNormShape(pNorm, 100-100*xC, 200-100*yC, 100*r);

    // d3.select('#templatePlot')
    //     .append("polygon")
    //         .attr("points", pointsToString(template))
    //         .attr("class", "guidePoint")
    //         .attr("id", "debugShape");
    ////// END DEBUG

    y = Math.pow(Math.pow(r, p) - Math.pow(t - xC, p), 1 / p) - yC;

    return 0.5 * y;
}

function maxDistanceShape(beta, p, w = 100, h = 100) {
    var numSegments = 201;
    var pts = [];

    if (p == Number.POSITIVE_INFINITY) {
        if (beta >= 1) {
            pts.push({ x: 0, y: h });
            pts.push({ x: 0, y: h - h * beta / 2. });
            pts.push({ x: w, y: h - h * beta / 2. });
            pts.push({ x: w, y: h });
        }
        else {
            // Since the square is not rounded, when beta < 1 the region
            // to be tested is always empty
            pts.push({ x: 0, y: h });
            pts.push({ x: w, y: h });
        }
    }
    else {
        for (var i = 0; i < numSegments; i++) {
            var t = i / (numSegments - 1);
            var y = maxDistance(t, beta, p);
            pts.push({ x: t * w, y: h - y * h });
        }
    }

    //Add the first point to the end to ensure a closed path
    if (closed) {
        pts.push({ x: pts[0].x, y: pts[0].y });
    }
    return pts;
}

function checkPointOnEdge(p, q, r, pNorm, beta) {
    var theta = Math.atan2(q.y - p.y, q.x - p.x);

    var t = dot([q.x - p.x, q.y - p.y], [r.x - p.x, r.y - r.x]);
    // var c = {x: 0.5*(p.x+q.x), y: 0.5*(p.y+q.y)};
    // var cR = translate(c, -x1, -y1);
    // cR = rotate(cR, -theta);
    // cR = translate(cR, x1, y1);

    var qR = { x: q.x, y: q.y };
    qR = translate(qR, -p.x, -p.y);
    qR = rotate(qR, -theta);
    qR = translate(qR, p.x, p.y);

    var rR = { x: r.x, y: r.y };
    translate(rR, -p.x, -p.y);
    rotate(rR, -theta);
    translate(rR, p.x, p.y);

    d_pq = lpNorm(p.x, p.y, qR.x, qR.y, p);
    d_pr = lpNorm(p.x, p.y, rR.x, rR.y, p);

    mt = maxDistance(t, beta, pNorm);
}