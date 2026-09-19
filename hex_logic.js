class HexState {
    constructor(size = 9) {
        this.size = size;
        this.board = new Array(size * size).fill(-1);
        this.currentPlayer = 0; // 0: Niebieski (Czarny z Pythona), 1: Czerwony
        this.isTerminal = false;
        this.movesPlayed = 0;
        this.firstMoveAction = null;
        this.swapAction = size * size; // Akcja nr 81 dla planszy 9x9
    }

    legalActions() {
        let actions = [];
        for (let i = 0; i < this.board.length; i++) {
            if (this.board[i] === -1) actions.push(i);
        }
        if (this.movesPlayed === 1) actions.push(this.swapAction);
        return actions;
    }

    applyAction(action) {
        if (this.isTerminal) return;

        if (action === this.swapAction) {
            this.board[this.firstMoveAction] = 1;
            this.currentPlayer = 0;
            this.movesPlayed++;
            return;
        }

        if (this.board[action] !== -1) return;

        this.board[action] = this.currentPlayer;
        if (this.movesPlayed === 0) this.firstMoveAction = action;

        if (this.checkWin(this.currentPlayer)) {
            this.isTerminal = true;
        } else {
            this.currentPlayer = 1 - this.currentPlayer;
        }
        this.movesPlayed++;
    }

    getNeighbors(action) {
        const r = Math.floor(action / this.size), c = action % this.size;
        const dirs = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]];
        let out = [];
        for (let [dr, dc] of dirs) {
            let nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < this.size && nc >= 0 && nc < this.size) {
                out.push(nr * this.size + nc);
            }
        }
        return out;
    }

    checkWin(player) {
        let visited = new Set();
        let stack = [];
        for (let i = 0; i < this.size; i++) {
            let node = (player === 0) ? i : i * this.size;
            if (this.board[node] === player) stack.push(node);
        }

        while (stack.length > 0) {
            let curr = stack.pop();
            if (visited.has(curr)) continue;
            visited.add(curr);
            
            let r = Math.floor(curr / this.size), c = curr % this.size;
            if ((player === 0 && r === this.size - 1) || (player === 1 && c === this.size - 1)) return true;

            for (let n of this.getNeighbors(curr)) {
                if (this.board[n] === player && !visited.has(n)) stack.push(n);
            }
        }
        return false;
    }
}
