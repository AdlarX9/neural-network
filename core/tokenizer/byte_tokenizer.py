from __future__ import annotations
from .tokenizer import Tokenizer
from collections import Counter
import heapq
import re


class ByteTokenizer(Tokenizer):
    _PRETOKENIZE_PATTERN = re.compile(r"\s+|[A-Za-zÀ-ÖØ-öø-ÿ]+|\d+|[^A-Za-zÀ-ÖØ-öø-ÿ\d\s]+", re.UNICODE)

    def __init__(self: ByteTokenizer, vocab_size: int = 32_768) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        self.vocab_size = vocab_size
        self.V: dict[tuple[int, ...], int] = {(i,): i for i in range(256)}
        self.R: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._token_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        self._merge_ranks: dict[tuple[int, int], int] = {}

    def _pretokenize(self: ByteTokenizer, text: str) -> list[bytes]:
        return [match.group(0).encode("utf-8") for match in self._PRETOKENIZE_PATTERN.finditer(text)]

    def build_vocab(self: ByteTokenizer, corpus: str) -> None:
        self.V = {(i,): i for i in range(256)}
        self.R = []
        self._token_bytes = [bytes([i]) for i in range(256)]
        self._merge_ranks = {}
        if not corpus:
            return
        if self.vocab_size == 256:
            return
        pieces = self._pretokenize(corpus)
        if not pieces:
            return

        # Give a strong priority to full pre-tokenized pieces that are exactly 2 bytes
        # so very common short words (for example "le", "la", "de" in French) are
        # learned early instead of being skipped by longer composite merges.
        forced_pair_bonus: Counter[tuple[int, int]] = Counter()
        for piece in pieces:
            if len(piece) == 2:
                forced_pair_bonus[(piece[0], piece[1])] += 1

        forced_pairs_top_n = 512
        forced_pair_bonus = Counter(dict(forced_pair_bonus.most_common(forced_pairs_top_n)))
        forced_pair_multiplier = 10_000

        def pair_priority_score(pair: tuple[int, int]) -> int:
            return pair_counts.get(pair, 0) + (forced_pair_bonus.get(pair, 0) * forced_pair_multiplier)

        sequences: list[list[int]] = []
        for piece in pieces:
            if piece:
                sequences.append(list(piece))
        if not sequences:
            return
        prev: list[list[int]] = []
        next_: list[list[int]] = []
        alive: list[bytearray] = []
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_occurrences: dict[
            tuple[int, int],
            set[tuple[int, int]],
        ] = {}
        for sequence_id, sequence in enumerate(sequences):
            length = len(sequence)
            sequence_prev = [i - 1 for i in range(length)]
            sequence_next = [i + 1 if i + 1 < length else -1 for i in range(length)]
            sequence_alive = bytearray(b"\x01") * length
            prev.append(sequence_prev)
            next_.append(sequence_next)
            alive.append(sequence_alive)
            for position in range(length - 1):
                pair = (sequence[position], sequence[position + 1])
                pair_counts[pair] += 1
                pair_occurrences.setdefault(pair, set()).add((sequence_id, position))
        heap = []
        for (left, right), count in pair_counts.items():
            pair = (left, right)
            score = count + (forced_pair_bonus.get(pair, 0) * forced_pair_multiplier)
            heap.append((-score, -count, left, right))
        heapq.heapify(heap)
        show_progress = len(corpus) >= 10_000
        next_token_id = 256
        target_merges = self.vocab_size - 256
        while next_token_id < self.vocab_size:
            merge_index = next_token_id - 256
            if show_progress and merge_index % 100 == 0:
                progress = merge_index / target_merges * 100
                print("Vocabulary progress:" f" {progress:.2f}%", end="\r")
            best_pair = None
            while heap:
                negative_score, negative_count, left, right = heapq.heappop(heap)
                pair = (left, right)
                current_count = pair_counts.get(pair, 0)
                current_score = pair_priority_score(pair)
                if (
                    current_score == -negative_score
                    and current_count == -negative_count
                    and current_count > 0
                ):
                    best_pair = pair
                    break
            if best_pair is None:
                break
            left, right = best_pair
            occurrences = pair_occurrences.get(best_pair)
            if not occurrences:
                pair_counts.pop(best_pair, None)
                continue
            merged_id = next_token_id
            merged_bytes = self._token_bytes[left] + self._token_bytes[right]
            self._token_bytes.append(merged_bytes)
            left_tuple = tuple(self._token_bytes[left])
            right_tuple = tuple(self._token_bytes[right])
            merged_tuple = tuple(merged_bytes)
            self.V[merged_tuple] = merged_id
            self.R.append((left_tuple, right_tuple))
            self._merge_ranks[best_pair] = merge_index
            next_token_id += 1

            occurrences_by_sequence: dict[int, list[int]] = {}
            for sequence_id, position in occurrences:
                occurrences_by_sequence.setdefault(sequence_id, []).append(position)

            for (
                sequence_id,
                positions,
            ) in occurrences_by_sequence.items():
                positions.sort()
                sequence = sequences[sequence_id]
                sequence_prev = prev[sequence_id]
                sequence_next = next_[sequence_id]
                sequence_alive = alive[sequence_id]
                for left_position in positions:
                    if not sequence_alive[left_position]:
                        continue
                    right_position = sequence_next[left_position]
                    if right_position == -1:
                        continue
                    if not sequence_alive[right_position]:
                        continue
                    if sequence[left_position] != left or sequence[right_position] != right:
                        continue
                    previous_position = sequence_prev[left_position]
                    next_position = sequence_next[right_position]

                    if previous_position != -1:
                        previous_id = sequence[previous_position]
                        old_pair = (previous_id, left)
                        pair_counts[old_pair] -= 1
                        old_occurrences = pair_occurrences.get(old_pair)
                        if old_occurrences is not None:
                            old_occurrences.discard((sequence_id, previous_position))

                    pair_counts[best_pair] -= 1
                    occurrences.discard((sequence_id, left_position))

                    if next_position != -1:
                        next_id = sequence[next_position]
                        old_pair = (right, next_id)
                        pair_counts[old_pair] -= 1
                        old_occurrences = pair_occurrences.get(old_pair)
                        if old_occurrences is not None:
                            old_occurrences.discard((sequence_id, right_position))

                    sequence[left_position] = merged_id
                    sequence_alive[right_position] = 0
                    sequence_next[left_position] = next_position
                    if next_position != -1:
                        sequence_prev[next_position] = left_position

                    if previous_position != -1:
                        new_pair = (sequence[previous_position], merged_id)
                        pair_counts[new_pair] += 1
                        pair_occurrences.setdefault(new_pair, set()).add((sequence_id, previous_position))
                        new_count = pair_counts[new_pair]
                        new_score = new_count + (forced_pair_bonus.get(new_pair, 0) * forced_pair_multiplier)
                        heapq.heappush(heap, (-new_score, -new_count, new_pair[0], new_pair[1]))

                    if next_position != -1:
                        new_pair = (merged_id, sequence[next_position])
                        pair_counts[new_pair] += 1
                        pair_occurrences.setdefault(new_pair, set()).add((sequence_id, left_position))
                        new_count = pair_counts[new_pair]
                        new_score = new_count + (forced_pair_bonus.get(new_pair, 0) * forced_pair_multiplier)
                        heapq.heappush(heap, (-new_score, -new_count, new_pair[0], new_pair[1]))
            if pair_counts.get(best_pair, 0) <= 0:
                pair_counts.pop(best_pair, None)
            if not pair_occurrences.get(best_pair):
                pair_occurrences.pop(best_pair, None)
        if show_progress:
            print("Vocabulary progress: 100.00%")
        if len(self.V) != self.vocab_size:
            raise RuntimeError(
                "BPE vocabulary construction "
                "stopped prematurely: "
                f"expected {self.vocab_size} "
                f"tokens, got {len(self.V)}."
            )
        if len(self._token_bytes) != self.vocab_size:
            raise RuntimeError(
                "BPE token byte table has an "
                "invalid size: "
                f"expected {self.vocab_size}, "
                f"got {len(self._token_bytes)}."
            )
        if len(self.R) != self.vocab_size - 256:
            raise RuntimeError(
                "BPE rule count is inconsistent: " f"expected {self.vocab_size - 256}, " f"got {len(self.R)}."
            )
        if len(self._merge_ranks) != (self.vocab_size - 256):
            raise RuntimeError(
                "BPE merge rank count is inconsistent: "
                f"expected {self.vocab_size - 256}, "
                f"got {len(self._merge_ranks)}."
            )
        if set(self.V.values()) != set(range(self.vocab_size)):
            raise RuntimeError("BPE vocabulary contains " "non-contiguous token IDs.")

    def tokenize(
        self: ByteTokenizer,
        text: str,
    ) -> list[int]:
        show = bool(len(text) > 100_000)
        if not text:
            return []
        pieces = self._pretokenize(text)
        result: list[int] = []
        for idx, piece in enumerate(pieces):
            if not piece:
                continue
            if show:
                progress = idx / len(pieces) * 100
                print("Tokenizing progress: " f"{progress:.2f}%", end="\r")
            sequence = list(piece)
            length = len(sequence)
            if length == 1:
                result.append(sequence[0])
                continue
            prev = [i - 1 for i in range(length)]
            next_ = [i + 1 if i + 1 < length else -1 for i in range(length)]
            alive = bytearray(b"\x01") * length
            heap = []
            for position in range(length - 1):
                pair = (
                    sequence[position],
                    sequence[position + 1],
                )
                rank = self._merge_ranks.get(pair)
                if rank is not None:
                    heapq.heappush(
                        heap,
                        (
                            rank,
                            position,
                        ),
                    )
            while heap:
                rank, left_position = heapq.heappop(heap)
                if not alive[left_position]:
                    continue
                right_position = next_[left_position]
                if right_position == -1:
                    continue
                if not alive[right_position]:
                    continue
                pair = (
                    sequence[left_position],
                    sequence[right_position],
                )
                current_rank = self._merge_ranks.get(pair)
                if current_rank != rank:
                    continue
                merged_id = self.V[tuple(self._token_bytes[pair[0]] + self._token_bytes[pair[1]])]
                previous_position = prev[left_position]
                next_position = next_[right_position]
                sequence[left_position] = merged_id
                alive[right_position] = 0
                next_[left_position] = next_position
                if next_position != -1:
                    prev[next_position] = left_position
                if previous_position != -1:
                    left_pair = (
                        sequence[previous_position],
                        sequence[left_position],
                    )
                    left_rank = self._merge_ranks.get(left_pair)
                    if left_rank is not None:
                        heapq.heappush(
                            heap,
                            (
                                left_rank,
                                previous_position,
                            ),
                        )
                if next_position != -1:
                    right_pair = (
                        sequence[left_position],
                        sequence[next_position],
                    )
                    right_rank = self._merge_ranks.get(right_pair)
                    if right_rank is not None:
                        heapq.heappush(
                            heap,
                            (
                                right_rank,
                                left_position,
                            ),
                        )
            position = 0
            while position != -1:
                if alive[position]:
                    result.append(sequence[position])
                position = next_[position]
        if show:
            print("Tokenizing progress: 100.00%")
        return result

    def untokenize(
        self: ByteTokenizer,
        tokens: list[int],
    ) -> str:
        byte_values = bytearray()
        for token_id in tokens:
            if token_id < 0 or token_id >= len(self._token_bytes):
                raise ValueError(f"Unknown token ID: " f"{token_id}")
            byte_values.extend(self._token_bytes[token_id])
        return bytes(byte_values).decode("utf-8")

    def get_data(self: ByteTokenizer) -> dict:
        data = {
            "vocab_size": self.vocab_size,
            "V": [[list(token), idx] for token, idx in self.V.items()],
            "R": [[list(left), list(right)] for left, right in self.R],
        }
        return data

    def load_from_data(self: ByteTokenizer, data: dict) -> None:
        self.vocab_size = data["vocab_size"]
        self.V = {tuple(token): idx for token, idx in data["V"]}
        self.R = [(tuple(left), tuple(right)) for left, right in data["R"]]
        self._token_bytes = []
        self._merge_ranks = {}

        max_token_id = max(
            self.V.values(),
            default=255,
        )
        self._token_bytes = [b"" for _ in range(max_token_id + 1)]
        for key, token_id in self.V.items():
            self._token_bytes[token_id] = bytes(key)

        for merge_index, (left, right) in enumerate(self.R):
            left_id = self.V[left]
            right_id = self.V[right]
            self._merge_ranks[(left_id, right_id)] = merge_index

    def create_file(self: ByteTokenizer) -> None:
        with open("./tokens.txt", "w", encoding="utf-8") as file:
            for token, idx in self.V.items():
                try:
                    file.write(str(idx) + " " * (6 - len(str(idx))) + bytes(token).decode("utf-8") + "\n")
                except:
                    file.write("not possible\n")

    def see_tokenization(self: ByteTokenizer, sentence: str) -> None:
        tokens = self.tokenize(sentence)
        for token in tokens:
            try:
                print(self._token_bytes[token].decode("utf-8"))
            except:
                print("not possible")
