package vader

import (
	"archive/zip"
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode"
)

func TestVader_PolarityScores(t *testing.T) {
	var languages []string
	dirs, err := os.ReadDir("lexicons")
	if err != nil {
		t.Fatal(err)
	}

	for _, dir := range dirs {
		if dir.IsDir() {
			languages = append(languages, dir.Name())
		}
	}

	for _, lang := range languages {

		v, err := NewVader(filepath.Join("lexicons", lang, lang+".zip"))
		if err != nil {
			t.Fatal(err)
		}

		testSentences, err := readTests(filepath.Join("lexicons", lang, "tests.tsv"))
		if err != nil {
			t.Fatal(err)
		}

		for index, tt := range testSentences {

			t.Run(fmt.Sprintf("%s - test #%d", lang, index), func(t *testing.T) {

				t.Log(tt.sentence)
				v.debug = func(format string, v ...interface{}) {
					t.Logf(format, v...)
				}

				got := v.PolarityScores(tt.sentence)

				t.Logf("current neg: %.3f, neu: %.3f, pos: %.3f, compound: %.3f", got.Negative, got.Neutral, got.Positive, got.Compound)
				t.Logf("expected neg: %.3f, neu: %.3f, pos: %.3f, compound: %.3f", tt.neg, tt.neu, tt.pos, tt.compound)

				t.Logf("sentiment: %s", got.Sentiment())

				if !inEpsilon(tt.pos, got.Positive, 0.01) {
					t.Fatalf("invalid positive, expected %.3f, got %.3f", tt.pos, got.Positive)
				}

				if !inEpsilon(tt.neg, got.Negative, 0.01) {
					t.Fatalf("invalid negative, expected %.3f, got %.3f", tt.neg, got.Negative)
				}

				if !inEpsilon(tt.neu, got.Neutral, 0.01) {
					t.Fatalf("invalid neutral, expected %.3f, got %.3f", tt.pos, got.Neutral)
				}

				if !inEpsilon(tt.compound, got.Compound, 0.01) {
					t.Fatalf("invalid compound, expected %.3f, got %.3f", tt.compound, got.Compound)
				}
			})
		}
	}
}

func BenchmarkPolarityScores(b *testing.B) {

	var languages []string
	dirs, err := os.ReadDir("lexicons")
	if err != nil {
		b.Fatal(err)
	}

	for _, dir := range dirs {
		if dir.IsDir() {
			languages = append(languages, dir.Name())
		}
	}

	for _, lang := range languages {

		v, err := NewVader(filepath.Join("lexicons", lang, lang+".zip"))
		if err != nil {
			b.Fatal(err)
		}

		testSentences, err := readTests(filepath.Join("lexicons", lang, "tests.tsv"))
		if err != nil {
			b.Fatal(err)
		}

		c := len(testSentences)
		b.Run("PolarityScores_"+lang, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				v.PolarityScores(testSentences[i%c].sentence)
			}
		})
	}
}

func Test_sequences(t *testing.T) {
	tests := []struct {
		input []string
		want  [][]string
	}{
		{[]string{"a"}, [][]string{{"a"}}},
		{[]string{"a", "*"}, [][]string{{"a", "*"}}},
		{[]string{"a", "*", "?"}, [][]string{{"a", "*"}, {"a", "*", "*"}}},
	}
	for index, tt := range tests {
		t.Run(fmt.Sprintf("test #%d", index), func(t *testing.T) {
			got := sequences(tt.input)

			if !reflect.DeepEqual(tt.want, got) {
				t.Fail()
			}
		})
	}
}

func TestFind(t *testing.T) {

	dummy, err := NewVader("lexicons/en/en.zip")
	if err != nil {
		t.Fatal(err)
	}

	root := &node{
		child: map[string]*node{
			"a": {
				isSet:   true,
				valence: 1,
				child: map[string]*node{
					"*": {
						child: map[string]*node{
							"z": {isSet: true, valence: 1.99},
						},
					},
					"a1": {isSet: true, valence: 1.1},
					"a2": {isSet: true, valence: 1.2},
				},
			},
			"b": {
				child: map[string]*node{},
				regex: map[*regexp.Regexp]*node{
					regexp.MustCompile(".*x"): {
						isSet:   true,
						valence: 2,
					},
				},
			},
		},
	}

	tests := []struct {
		input     []string
		matched   []string
		backwards bool
		pos       int
		want      bool
		value     float64
	}{
		{
			input:     []string{"a1", "a"},
			backwards: true,
			pos:       1,
			want:      true,
			value:     1.1,
		},
		{
			input:     []string{"a1", "a", "a2"},
			backwards: false,
			pos:       1,
			want:      true,
			value:     1.2,
		},
		{
			input:     []string{"a1", "a", "any", "z"},
			backwards: false,
			pos:       1,
			want:      true,
			value:     1.99,
		},
		{
			input:     []string{"a1", "b", "az", "k"},
			backwards: false,
			pos:       1,
			want:      false,
		},
		{
			input:     []string{"a1", "b", "ax", "k"},
			backwards: false,
			pos:       1,
			want:      true,
			value:     2,
		},
		{
			input:     []string{"ax", "b", "c", "d"},
			backwards: true,
			pos:       1,
			want:      true,
			value:     2,
		},
	}
	for i, tt := range tests {
		t.Run(fmt.Sprintf("test #%d", i), func(t *testing.T) {
			match, ok := root.find(dummy, tt.input, tt.pos, tt.backwards, tt.matched)

			if tt.want != ok {
				t.Fatal("invalid exists flag")
			}
			if tt.want {
				if match == nil {
					t.Fatal("match is nil")
				}

				if !inEpsilon(tt.value, match.valence, 0.001) {
					t.Fatal("invalid value")
				}
			} else if match != nil {
				t.Fatal("expected nil")
			}
		})
	}

}

func Test_trimPunctuation(t *testing.T) {
	v, err := NewVader("lexicons/en/en.zip")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		input string
		want  string
	}{
		{"", ""},
		{"...", "..."},
		{"ab.", "ab"},
		{".abc", "abc"},
		{":)", ":)"},
		{":D", ":D"},
	}
	for index, tt := range tests {
		t.Run(fmt.Sprintf("test #%d", index), func(t *testing.T) {
			got := v.trimPunctuation(tt.input)

			if tt.want != got {
				t.Fatal("invalid result")
			}

		})
	}
}

type testSentence struct {
	sentence                string
	pos, neg, neu, compound float64
}

func readTests(file string) (sentences []testSentence, err error) {

	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewScanner(f)
	for r.Scan() {
		token := strings.Split(r.Text(), "\t")
		if len(token) == 5 {

			s := testSentence{
				sentence: token[0],
			}

			s.neg, err = strconv.ParseFloat(token[1], 64)
			if err != nil {
				return nil, err
			}

			s.neu, err = strconv.ParseFloat(token[2], 64)
			if err != nil {
				return nil, err
			}

			s.pos, err = strconv.ParseFloat(token[3], 64)
			if err != nil {
				return nil, err
			}

			s.compound, err = strconv.ParseFloat(token[4], 64)
			if err != nil {
				return nil, err
			}

			sentences = append(sentences, s)
		}
	}
	return sentences, nil
}

func inEpsilon(a, b, epsilon float64) bool {
	if a == b {
		return true
	}
	return (a-b) < epsilon && (b-a) < epsilon
}

func Test_isUpper(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"a", false},
		{"AAa", false},
		{"A", true},
		{"AAAAAA", true},
	}
	for index, tt := range tests {
		t.Run(fmt.Sprintf("test #%d", index), func(t *testing.T) {
			got := isUpper(tt.input)

			if tt.want != got {
				t.Fail()
			}
		})
	}
}

func Test_splitWords(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"", []string{}},
		{"a b", []string{"a", "b"}},
		{"a b...", []string{"a", "b..."}},
		{"😀a b", []string{"😀", "a", "b"}},
		{"😀😀😀.", []string{"😀", "😀", "😀", "."}},
	}
	for index, tt := range tests {
		t.Run(fmt.Sprintf("test #%d", index), func(t *testing.T) {
			got := splitWords(tt.input)

			if !reflect.DeepEqual(tt.want, got) {
				t.Fail()
			}
		})
	}
}

func TestExample(t *testing.T) {
	vader, err := NewVader("lexicons/en/en.zip")
	if err != nil {
		panic(err)
	}

	score := vader.PolarityScores("VADER is smart, handsome, and funny!")

	fmt.Printf(
		"neg: %.3f, neu: %.3f, pos: %.3f, compound: %.3f (%s)",
		score.Negative,
		score.Neutral,
		score.Positive,
		score.Compound,
		score.Sentiment(),
	)
}

func TestErrors(t *testing.T) {

	var files []string

	t.Cleanup(func() {
		for _, file := range files {
			_ = os.Remove(file)
		}
	})

	v, err := NewVader("")
	if err == nil {
		t.Fatal("should fail with a empty string")
	} else if v != nil {
		t.Fatal("should returned a nil value")
	}

	file, err := makeTestLexicon(t, "", "a", "b")
	if err != nil {
		t.Fatal(err)
	}
	files = append(files, file)
	v, err = NewVader(file)
	if err == nil {
		t.Fatal("should return an error with a lexicon without emoji file.")
	} else if v != nil {
		t.Fatal("should returned a nil value")
	}

	file, err = makeTestLexicon(t, "a", "", "b")
	if err != nil {
		t.Fatal(err)
	}
	files = append(files, file)
	v, err = NewVader(file)
	if err == nil {
		t.Fatal("should return an error with a lexicon without language file.")
	} else if v != nil {
		t.Fatal("should returned a nil value")
	}

	file, err = makeTestLexicon(t, "a", "b", "")
	if err != nil {
		t.Fatal(err)
	}
	files = append(files, file)
	v, err = NewVader(file)
	if err == nil {
		t.Fatal("should return an error with a lexicon without lexicon file.")
	} else if v != nil {
		t.Fatal("should returned a nil value")
	}

	// invalid regexp
	file, err = makeTestLexicon(
		t,
		"a",
		"rule	<	*	-0.74	0	/[[/ * ?\n",
		"c",
	)
	if err != nil {
		t.Fatal(err)
	}
	files = append(files, file)
	v, err = NewVader(file)
	if err == nil {
		t.Fatal("should return an error with an invalid regexp.")
	} else if v != nil {
		t.Fatal("should returned a nil value")
	}

	// already declared value
	file, err = makeTestLexicon(
		t,
		"a",
		"rule	<	*	-0.74	0	a\nrule	<	*	-0.74	0	a\n",
		"c",
	)
	if err != nil {
		t.Fatal(err)
	}
	files = append(files, file)
	v, err = NewVader(file)
	if err == nil {
		t.Fatal("should return an error with a duplicated value.")
	} else if v != nil {
		t.Fatal("should returned a nil value")
	}

	// test invalid file
	v, err = NewVader("lexicons/test/emojis.tsv")
	if err == nil {
		t.Fatal("should return an error with a invalid file.")
	} else if v != nil {
		t.Fatal("should returned a nil value")
	}
}

func makeTestLexicon(t *testing.T, emojis, language, lexicon string) (string, error) {
	t.Helper()

	filename := filepath.Join(os.TempDir(), fmt.Sprintf("vader_lexicon_%d.tmp", time.Now().UnixNano()))

	f, err := os.Create(filename)
	if err != nil {
		return "", err
	}

	files := make(map[string]string)
	if emojis != "" {
		files["emojis.tsv"] = emojis
	}
	if language != "" {
		files["language.txt"] = language
	}
	if lexicon != "" {
		files["lexicon.tsv"] = lexicon
	}

	lex := zip.NewWriter(f)

	for name, content := range files {
		if entry, err := lex.Create(name); err != nil {
			return "", err
		} else if _, err = entry.Write([]byte(content)); err != nil {
			return "", err
		}
	}

	if err = lex.Close(); err != nil {
		return "", err
	}

	return filename, nil
}

func Test_contains(t *testing.T) {

	if !contains("abc", unicode.IsLetter) {
		t.Fatal("expected true")
	}

	if contains("abc", unicode.IsDigit) {
		t.Fatal("expected false")
	}
}
