package vader

import (
	"archive/zip"
	"bufio"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

var normText = transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)

// Scores encapsulates a single sentiment measure for a statement
type Scores struct {
	Negative float64
	Neutral  float64
	Positive float64
	Compound float64
}

func (s *Scores) Sentiment() string {
	if s.Compound >= 0.5 {
		return "positive"
	} else if s.Compound <= -0.05 {
		return "negative"
	}
	return "neutral"
}

type Vader struct {
	lexicon          map[string]float64
	emojis           map[string]float64
	replace          map[string]string
	lang             string
	capIncrement     float64
	removeDiacritics bool
	backward         *node
	forward          *node
	last             *node
	debug            func(format string, v ...interface{})
}

type node struct {
	child      map[string]*node
	regex      map[*regexp.Regexp]*node
	isSet      bool
	valence    float64
	mode       rune
	stop       bool
	count      int
	token      string
	expr       string
	exceptions [][]string
}

func (n *node) find(v *Vader, words []string, pos int, backwards bool, alreadyMatched []string) (child *node, found bool) {

	if pos == -1 {
		if n.isSet && !sliceContains(alreadyMatched, n.expr) {
			return n, true
		} else {
			return nil, false
		}
	} else if pos == len(words) {
		if n.isSet && !sliceContains(alreadyMatched, n.expr) {
			return n, true
		}
		return nil, false
	}

	var nextPos int
	if backwards {
		nextPos = pos - 1
	} else {
		nextPos = pos + 1
	}

	if n.regex != nil {
		for re, n := range n.regex {
			if !sliceContains(alreadyMatched, n.expr) && re.MatchString(words[pos]) {
				child = n
				found = true
				goto nextException
			}
		}
	}

	if !found {
		for _, search := range []string{words[pos], "*", "!"} {
			if child, found = n.child[search]; found {
				child, found = child.find(v, words, nextPos, backwards, alreadyMatched)
			}
			if found {
				goto nextException
			}
		}
	}

	if !found {
		if n.isSet && !sliceContains(alreadyMatched, n.expr) {
			child = n
		} else {
			return nil, false
		}
	}

nextException:
	for _, exception := range child.exceptions {
		for i, e := range exception {
			var c int

			if backwards {
				c = pos - ((len(exception) - 1) - i)
			} else {
				c = pos + i
			}

			if c < 0 || c > len(words)-1 {
				continue nextException
			} else if e == "*" {
				continue
			} else if words[c] != e {
				continue nextException
			}
		}
		// found a exception
		return nil, false
	}

	return child, true
}

// Creates a new Vader instance using the zip lexicon file.
func NewVader(vaderZipFile string) (*Vader, error) {

	f, err := zip.OpenReader(vaderZipFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	v := &Vader{
		lang:             "und",
		removeDiacritics: false,
		capIncrement:     0.733,
		debug:            func(format string, v ...interface{}) {},
		forward: &node{
			child: make(map[string]*node),
		},
		backward: &node{
			child: make(map[string]*node),
		},
		last: &node{
			child: make(map[string]*node),
		},
	}

	foundLexicon := false
	foundLanguage := false
	foundEmojis := false

	for _, file := range f.File {
		z, err := file.Open()
		if err != nil {
			return nil, err
		}
		switch file.Name {
		case "lexicon.tsv":
			foundLexicon = true
			v.readLexicon(z)
		case "language.txt":
			foundLanguage = true
			if err = v.readLanguage(z); err != nil {
				return nil, err
			}
		case "emojis.tsv":
			foundEmojis = true
			v.readEmojis(z)

		}
		z.Close()
	}

	if !foundLexicon {
		return nil, errors.New("unable to find the lexicon.tsv file")
	}
	if !foundLanguage {
		return nil, errors.New("unable to find the language.txt file")
	}
	if !foundEmojis {
		return nil, errors.New("unable to find the emojis.tsv file")
	}

	// avoid matching everything
	v.backward.isSet = false
	v.forward.isSet = false

	return v, nil
}

// Return a float for sentiment strength based on the input text. Positive values are positive valence, negative value are negative valence.
func (v *Vader) PolarityScores(sentence string) Scores {
	if v.removeDiacritics {
		if output, _, e := transform.String(normText, sentence); e == nil {
			sentence = output
		}
	}

	sentence = strings.TrimSpace(sentence)

	var sentiments []float64

	var words []string
	var lowers []string

	for _, token := range splitWords(sentence) {
		token = v.trimPunctuation(token)

		if v.replace != nil {
			if s, ok := v.replace[token]; ok {
				token = s
			}
		}

		words = append(words, token)
		lowers = append(lowers, strings.ToLower(token))
	}

	isCapDone := false
	isCapDiff := false // one or more words are capitalized, but not all of tem
	for i, word := range words {
		if !isCapDone && !isCapDiff && isUpper(word) {
			isCapDiff = contains(sentence, unicode.IsLower)
			isCapDone = true
		}

		valence := 0.0

		if x, ok := v.lexicon[lowers[i]]; ok {
			valence = x
			v.debug("#%d lexicon for %q = %.3f", i, word, x)
		} else if x, ok = v.emojis[words[i]]; ok {
			valence = x
			v.debug("#%d emoji for %s = %.3f", i, word, x)
		}
		if valence != 0.0 {
			matched := make([]string, 0)
			for {
				if n, ok := v.forward.find(v, lowers, i, false, matched); ok {

					matched = append(matched, n.expr)
					switch n.mode {
					case '=':
						valence = n.valence
					case '-':
						valence -= n.valence
					case '+':
						valence += n.valence
					case '*':
						valence *= n.valence
					}

					v.debug("#%d found forward %s [valence %s %.3f = %.3f]\n", i, n.expr, string(n.mode), n.valence, valence)

					if n.stop {
						break
					}
					continue
				}
				break
			}
			matched = make([]string, 0)
			for {
				if n, ok := v.backward.find(v, lowers, i, true, matched); ok {
					matched = append(matched, n.expr)

					switch n.mode {
					case '=':
						valence = n.valence
					case '-':
						if valence < 0 {
							valence += n.valence
						} else {
							valence -= math.Abs(n.valence)
						}
					case '+':
						if valence < 0 {
							valence -= math.Abs(n.valence)
						} else {
							valence += n.valence
						}
					case '*':
						valence *= n.valence
					}

					v.debug("#%d found backward %s [valence %s %.3f = %.3f]\n", i, n.expr, string(n.mode), n.valence, valence)

					if isCapDiff && isUpper(words[i-(n.count-1)]) {
						if valence > 0 {
							valence += v.capIncrement
						} else {
							valence -= v.capIncrement
						}
					}

					if n.stop {
						break
					}
					continue
				}
				break
			}

			if isCapDiff && isUpper(word) {
				if valence > 0 {
					valence += v.capIncrement
				} else {
					valence -= v.capIncrement
				}
			}
		}

		v.debug("-> %.3f (#%d %s) \n", valence, i, lowers[i])

		sentiments = append(sentiments, valence)
	}

	for pos := range lowers {
		if last, ok := v.last.find(v, lowers, pos, true, []string{}); ok {
			for i := range lowers {
				if i < pos {
					sentiments[i] *= 1 - last.valence
				} else if i > pos {
					sentiments[i] *= 1 + last.valence
				}
			}
			break
		}
	}

	return v.score(sentiments, sentence)
}

func (v *Vader) score(sentiments []float64, sentence string) Scores {
	score := Scores{
		Negative: 0,
		Neutral:  0,
		Positive: 0,
		Compound: 0,
	}

	if len(sentiments) > 0 {
		sum := 0.0

		pos := 0.0
		neg := 0.0
		neu := 0.0

		for _, s := range sentiments {
			sum += s

			if s > 0 {
				pos += s + 1
			} else if s < 0 {
				neg += s - 1
			} else {
				neu += 1
			}
		}

		emphasis := v.punctuationEmphasis(sentence)
		if sum > 0 {
			sum += emphasis
		} else if sum < 0 {
			sum -= emphasis
		}

		score.Compound = normalize(sum, 15)

		if pos > math.Abs(neg) {
			pos += emphasis
		} else if pos < math.Abs(neg) {
			neg -= emphasis
		}

		total := pos + math.Abs(neg) + neu

		score.Positive = roundTo(math.Abs(pos/total), 3)
		score.Negative = roundTo(math.Abs(neg/total), 3)
		score.Neutral = roundTo(math.Abs(neu/total), 3)

	}
	return score
}

func (v *Vader) punctuationEmphasis(sentence string) float64 {
	amplify := 0.0

	count := float64(strings.Count(sentence, "!"))
	if count > 4 {
		count = 4
	}
	amplify = count * 0.292

	count = float64(strings.Count(sentence, "?"))
	if count > 3 {
		amplify += 0.96
	} else if count > 0 {
		amplify += 0.18 * count
	}

	return amplify
}

func (v *Vader) readEmojis(r io.ReadCloser) {
	s := bufio.NewScanner(r)

	v.emojis = make(map[string]float64)
	for s.Scan() {
		token := strings.Split(s.Text(), "\t")
		if len(token) >= 2 {
			f, err := strconv.ParseFloat(token[1], 64)
			if err != nil {
				continue
			}
			v.emojis[token[0]] = f
		}
	}
}

func (v *Vader) readLexicon(r io.ReadCloser) {
	s := bufio.NewScanner(r)

	v.lexicon = make(map[string]float64)
	for s.Scan() {
		token := strings.Split(s.Text(), "\t")
		if len(token) >= 2 {
			f, err := strconv.ParseFloat(token[1], 64)
			if err != nil {
				continue
			}
			v.lexicon[token[0]] = f
		}
	}
}

func (v *Vader) readLanguage(r io.ReadCloser) error {
	s := bufio.NewScanner(r)
	for s.Scan() {
		token := strings.Split(s.Text(), "\t")
		if len(token) >= 2 {
			switch strings.TrimSpace(token[0]) {
			case "#": // comment
				continue
			case "lang":
				v.lang = token[1]
			case "cap_increment":
				if f, err := strconv.ParseFloat(token[1], 64); err == nil {
					v.capIncrement = f
				}
			case "remove_diacritics":
				v.removeDiacritics = strings.EqualFold(token[1], "true")
			case "replace":
				if len(token) != 3 {
					continue
				}
				if v.replace == nil {
					v.replace = map[string]string{}
				}
				v.replace[token[1]] = token[2]
			case "rule":
				// rule [tab] mode [tab] valence [tab] precedence [tab] stop [tab] sequence [[tab] exception...]
				if len(token) < 6 {
					continue
				}

				var typ rune
				switch token[1][0] {
				case '<', '>', '@':
					typ = rune(token[1][0])
				default:
					continue
				}

				var mode rune
				switch token[2][0] {
				case '=', '*', '-', '+':
					mode = rune(token[2][0])
				default:
					continue
				}

				f, err := strconv.ParseFloat(token[3], 64)
				if err != nil {
					continue
				}

				stop := token[4] == "1"

				entries := sequences(strings.Fields(token[5]))

				for _, sequence := range entries {
					var cur *node
					switch typ {
					case '<':
						cur = v.backward
					case '>':
						cur = v.forward
					case '@':
						cur = v.last
					}

					for _, word := range reversible(sequence, typ == '<') {
						if strings.HasPrefix(word, "/") && strings.HasSuffix(word, "/") && len(word) > 3 {
							// regex
							expr := word[1 : len(word)-2]
							e, err := regexp.Compile(expr)
							if err != nil {
								return fmt.Errorf("invalid regexp expression: %s", expr)
							}

							if cur.regex == nil {
								cur.regex = make(map[*regexp.Regexp]*node)
							}
							n := &node{
								child: make(map[string]*node),
								token: word,
							}
							cur.regex[e] = n
							cur = n

							continue
						} else if r, ok := cur.child[word]; ok {
							cur = r
						} else {
							cur.child[word] = &node{
								child: make(map[string]*node),
							}
							cur = cur.child[word]
							cur.token = word
						}
					}
					if cur.isSet {
						return fmt.Errorf("the sequence is already defined: %s", s.Text())
					} else {
						cur.isSet = true
						cur.valence = f
						cur.mode = mode
						cur.stop = stop
						cur.count = len(sequence)
						cur.expr = strings.Join(sequence, " ")
						if len(token) > 6 {
							cur.exceptions = make([][]string, 0)
							for _, exception := range token[6:] {
								cur.exceptions = append(cur.exceptions, strings.Fields(exception))
							}
						}
					}
				}
			}
		}
	}
	return nil
}

func (v *Vader) trimPunctuation(s string) string {
	if _, ok := v.emojis[s]; ok {
		return s
	}
	if v := strings.TrimFunc(s, unicode.IsPunct); v != "" {
		return v
	}
	return s
}

func sequences(s []string) (r [][]string) {
	var n []string
	for i := 0; i < len(s); i++ {
		if s[i] == "?" {
			if len(n) > 0 {
				r = append(r, n)
			}
			n = append(n, "*")
		} else {
			n = append(n, s[i])
		}
	}
	r = append(r, n)
	return
}

func contains(s string, f func(rune) bool) bool {
	for _, r := range s {
		if f(r) {
			return true
		}
	}
	return false
}

// Normalize the score to be between -1 and 1 using an alpha that approximates the max expected value
func normalize(score, alpha float64) float64 {
	normScore := score / math.Sqrt((score*score)+alpha)
	if normScore < -1.0 {
		return -1.0
	} else if normScore > 1.0 {
		return 1.0
	}
	return normScore
}

func roundTo(n float64, decimals uint32) float64 {
	return math.Round(n*math.Pow(10, float64(decimals))) / math.Pow(10, float64(decimals))
}
func isUpper(text string) bool {
	for _, r := range text {
		if !unicode.IsUpper(r) {
			return false
		}
	}
	return true
}

func sliceContains(items []string, item string) bool {
	for _, n := range items {
		if n == item {
			return true
		}
	}
	return false
}

func splitWords(sentence string) []string {
	words := make([]string, 0)
	buf := strings.Builder{}
	for _, r := range sentence {
		if unicode.IsSpace(r) {
			if buf.Len() > 0 {
				words = append(words, buf.String())
				buf.Reset()
			}
		} else if unicode.IsSymbol(r) {
			words = append(words, string(r))
		} else {
			buf.WriteRune(r)
		}
	}
	if buf.Len() > 0 {
		words = append(words, buf.String())
	}
	return words
}

func reversible(s []string, reverse bool) []string {
	if !reverse {
		return s
	}
	r := make([]string, len(s))
	copy(r, s)
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return r
}
