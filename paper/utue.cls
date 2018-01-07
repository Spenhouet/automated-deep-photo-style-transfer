\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{utue}[2011/11/26 University of Tuebingen Corporate
Design for Computer Graphics]

\LoadClass[10pt,a4paper,twocolumn]{article}

\RequirePackage{graphicx,ae}
\RequirePackage{color}

%Define uni tuebingen corporate color
\definecolor{ecs100}{RGB}{161,22,58}

%Define optional subtitle
\def\@subtitle{\relax}
\newcommand{\subtitle}[1]{\gdef\@subtitle{#1}}

\newcommand*{\@teaser}{}
\newcommand*{\teaser}[1]{\renewcommand*{\@teaser}{#1}}

\renewcommand{\maketitle}{
  %Generates titlepage
  \thispagestyle{empty}
  \onecolumn
  \hfill\includegraphics[width=0.45\columnwidth]{images/tuebingen4b_um_rot_small.pdf}\\
  \vspace*{40ex}
  \begin{center}
    \begingroup
    \sffamily\scshape
    \LARGE\textbf \@title\\
    \vspace*{2ex}
    \Large \@author\\
    \vspace*{1ex}
    \normalsize \@date
    \endgroup
  \end{center}  
  \vfill   
  \colorbox{ecs100}{
    \hspace*{0.50\linewidth}
    \parbox{.50\linewidth}{
      \textcolor{white}
      {  
        \begingroup
        \sffamily\scshape
        \raggedleft \if\@subtitle\relax\else\Large\@subtitle\\\fi
        \raggedleft \normalsize Practical Course on Computer Graphics\\
        \endgroup
      }
    }
  }

  %Adds an empty page
  \clearpage
  \thispagestyle{empty}
  \mbox{}
  \twocolumn

  %Generates title with teaser if existing
  \clearpage
  \setcounter{page}{1}
  \twocolumn[
  \vspace*{3ex}
  \begin{center}
    \begingroup\LARGE\sffamily\bfseries\scshape\@title\endgroup\\
    \vspace*{3ex}
    \normalsize \@author
    \if\@teaser\relax\else
    \begingroup
    \vspace*{5ex} 
    \def\@captype{figure}%
    \noindent\@teaser
    \endgroup\par
    \fi
  \end{center}
  \vspace*{3ex}
  ]
}

%Defines sectionfont-styles
\newcommand{\utue@sectionfont}{}
\newcommand{\sectionfont}[1]{\renewcommand{\utue@sectionfont}{#1}}
\renewcommand\section{\@startsection{section}{1}{\z@}%
                {-2ex \@plus -1ex \@minus -.2ex}%
                {0.8ex \@plus .2ex}%
                {\reset@font\normalsize\sffamily\bfseries\scshape\utue@sectionfont}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                {-1.8ex\@plus -1ex \@minus -.2ex}%
                {0.8ex \@plus .2ex}%
                {\reset@font\normalsize\sffamily\bfseries\utue@sectionfont}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                {-1.8ex\@plus -1ex \@minus -.2ex}%
                {0.8ex \@plus .2ex}%
                {\reset@font\sffamily\normalsize\utue@sectionfont}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{0em}%
                {1ex \@plus 1ex \@minus.2ex}%
                {-1em}%
                {\reset@font\normalsize\sffamily\bfseries\utue@sectionfont}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                {3.25ex \@plus1ex \@minus .2ex}%
                {-1em}%
                {\reset@font\normalsize\sffamily\bfseries\utue@sectionfont}}

