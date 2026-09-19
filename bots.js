class HeuristicBot {
    constructor(size = 9) {
        this.size = size;
        this.fairOpenings = this.generateFairOpenings();
    }

    generateFairOpenings() {
        let openings = [];
        let center = Math.floor(this.size / 2);
        for (let r = 1; r < this.size - 1; r++) {
            for (let c = 1; c < this.size - 1; c++) {
                let dist = Math.abs(r - center) + Math.abs(c - center);
                if (dist === 2 || dist === 3) openings.push(r * this.size + c);
            }
        }
        return openings;
    }

    step(game) {
        const legal = game.legalActions();
        if (legal.length === 0) return null;

        // 0. Zasada otwarcia
        if (game.movesPlayed === 0) {
            if (Math.random() < 0.25) {
                let center = Math.floor(this.size / 2);
                let centerMoves = [];
                for (let r = center - 2; r <= center + 2; r++) {
                    for (let c = center - 2; c <= center + 2; c++) {
                        let move = r * this.size + c;
                        if (legal.includes(move)) centerMoves.push(move);
                    }
                }
                if (centerMoves.length > 0) return centerMoves[Math.floor(Math.random() * centerMoves.length)];
            }
            let validOpenings = this.fairOpenings.filter(m => legal.includes(m));
            if (validOpenings.length > 0) return validOpenings[Math.floor(Math.random() * validOpenings.length)];
        }

        if (game.movesPlayed === 1 && legal.includes(game.swapAction)) {
            let r = Math.floor(game.firstMoveAction / this.size), c = game.firstMoveAction % this.size;
            let center = Math.floor(this.size / 2);
            if (Math.abs(r - center) <= 2 && Math.abs(c - center) <= 2) return game.swapAction;
        }

        const player = game.currentPlayer;
        const opponent = 1 - player;
        
        const botDistStart = this.getDistances(game, player, true);
        const botDistEnd = this.getDistances(game, player, false);
        const oppDistStart = this.getDistances(game, opponent, true);
        const oppDistEnd = this.getDistances(game, opponent, false);

        let bestScore = null;
        let bestActions = [];

        for (let action of legal) {
            if (action === game.swapAction) continue;
            
            let botPath = botDistStart[action] + botDistEnd[action];
            let oppPath = oppDistStart[action] + oppDistEnd[action];
            
            let criticalScore = Math.min(botPath, oppPath + 0.1);
            let totalPaths = botPath + oppPath;
            let r = Math.floor(action / this.size), c = action % this.size;
            let centerDist = Math.abs(r - Math.floor(this.size / 2)) + Math.abs(c - Math.floor(this.size / 2));
            
            let score = [criticalScore, totalPaths, centerDist];
            
            if (bestScore === null || this.isBetter(score, bestScore)) {
                bestScore = score;
                bestActions = [action];
            } else if (score[0] === bestScore[0] && score[1] === bestScore[1] && score[2] === bestScore[2]) {
                bestActions.push(action);
            }
        }
        return bestActions.length > 0 ? bestActions[Math.floor(Math.random() * bestActions.length)] : legal[0];
    }

    isBetter(a, b) {
        if (a[0] !== b[0]) return a[0] < b[0];
        if (a[1] !== b[1]) return a[1] < b[1];
        return a[2] < b[2];
    }

    getDistances(game, player, isStartEdge) {
        const size = this.size;
        let dist = new Array(size * size).fill(Infinity);
        let q = [];
        for (let i = 0; i < size; i++) {
            let node = (player === 0) ? (isStartEdge ? i : (size - 1) * size + i) : (isStartEdge ? i * size : i * size + (size - 1));
            if (game.board[node] === player) { dist[node] = 0; q.unshift(node); }
            else if (game.board[node] === -1) { dist[node] = 1; q.push(node); }
        }
        while (q.length > 0) {
            let curr = q.shift(), d = dist[curr];
            for (let n of game.getNeighbors(curr)) {
                if (game.board[n] === 1 - player) continue;
                let cost = (game.board[n] === player) ? 0 : 1;
                if (d + cost < dist[n]) {
                    dist[n] = d + cost;
                    if (cost === 0) q.unshift(n); else q.push(n);
                }
            }
        }
        return dist;
    }
}

// Funkcja obsługująca model PPO z ONNX
async function getPPOMove(game, session) {
    const size = game.size;
    const p = game.currentPlayer;
    const rot = Math.random() < 0.5; // Opcjonalna losowa rotacja planszy
    
    // Budowanie tensora 3x9x9 (Kanały: my, opp, empty)
    const data = new Float32Array(3 * size * size);
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            let rr = rot ? (size - 1 - r) : r;
            let rc = rot ? (size - 1 - c) : c;
            if (p === 1) { let tmp = rr; rr = rc; rc = tmp; }
            let val = game.board[rr * size + rc];
            let idxBase = r * size + c;
            
            if (val === p) data[0 * 81 + idxBase] = 1.0;
            else if (val === 1 - p) data[1 * 81 + idxBase] = 1.0;
            else if (val === -1) data[2 * 81 + idxBase] = 1.0;
        }
    }
    
    const tensor = new ort.Tensor('float32', data, [1, 3, size, size]);
    const results = await session.run({ obs: tensor });
    const logits = results.logits.data; // Tablica 82 elementów
    
    let bestAction = -1;
    let bestLogit = -Infinity;
    const legal = new Set(game.legalActions());
    
    // Szukanie najwyższego logitu tylko w legalnych akcjach
    for (let a = 0; a <= size * size; a++) {
        let mapped = a;
        if (a !== game.swapAction) {
            let r = Math.floor(a / size), c = a % size;
            if (rot) { r = size - 1 - r; c = size - 1 - c; }
            if (p === 1) { let tmp = r; r = c; c = tmp; }
            mapped = r * size + c;
        }
        if (legal.has(mapped) && logits[a] > bestLogit) {
            bestLogit = logits[a];
            bestAction = mapped;
        }
    }
    return bestAction;
}