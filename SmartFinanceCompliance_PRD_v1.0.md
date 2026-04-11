# Smart Finance Compliance
## Product Requirements Document

| | |
|---|---|
| **Document Version** | v1.0 — Initial Release |
| **Date** | June 2025 |
| **Product Type** | B2B SaaS — RAG-based Compliance Platform |
| **Target Market** | Nepal — CA Firms, BFSI Sector, Large Corporates |
| **Status** | Pre-Seed / Prototype Phase |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Success Metrics](#3-goals--success-metrics)
4. [Target Users & Market](#4-target-users--market)
5. [Solution Overview & Feature Requirements](#5-solution-overview--feature-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Competitive Analysis](#7-competitive-analysis)
8. [Go-to-Market Strategy](#8-go-to-market-strategy)
9. [Development Milestones & Roadmap](#9-development-milestones--roadmap)
10. [Risks & Mitigations](#10-risks--mitigations)
11. [Cost & Resource Estimates](#11-cost--resource-estimates)
12. [Open Questions & Dependencies](#12-open-questions--dependencies)
13. [Appendix: Glossary](#13-appendix-glossary)

---

## 1. Executive Summary

Smart Finance Compliance is a Retrieval-Augmented Generation (RAG)-based SaaS platform designed to eliminate the compliance research bottleneck faced by Chartered Accountants, financial professionals, and legal compliance teams across Nepal. The platform provides instant, cited, and hierarchy-aware answers sourced directly from the Nepal Gazette, NRB directives, IRD circulars, and other authoritative government publications.

Currently, a qualified CA spending 4 hours researching a single tax or regulatory question is the industry norm — not the exception. Smart Finance Compliance reduces that to under 5 seconds, with traceable source citations, bilingual (Nepali/English) synthesis, and real-time regulatory updates.

| Metric | Value |
|---|---|
| ⏱ Research time per query | **4 hours → 5 seconds** |
| 📈 Estimated Nepal TAM/year | **रू 25–40 Crore (~USD $2–3M)** |
| 💰 Post-setup profit margin | **~90%** |

---

## 2. Problem Statement

Nepal's financial and legal compliance ecosystem suffers from a fundamental information architecture failure. Government regulations — Acts, Rules, Finance Act amendments, and regulatory circulars — are published as scanned, image-based PDFs in the Nepal Gazette. This creates a compounding set of problems for professionals who must act on them.

### 2.1 Core Pain Points

#### Pain Point 1: The "Ctrl+F" Void
The vast majority of government laws are published as scanned, image-based PDFs. Keyword search is impossible. A professional must read every page manually to find a specific provision.

#### Pain Point 2: The Regulatory Hierarchy Trap
An NRB directive or IRD circular can silently override a principal Act overnight. Missing a recent circular means advising clients based on superseded law — a direct liability risk. Professionals have no systematic way to track these "floating overrides."

#### Pain Point 3: The Table Nightmare
Tax rates, penalty structures, and exemption schedules are buried in complex multi-column tables within scanned PDFs. Standard OCR tools and general-purpose AI models scramble these figures, producing dangerously inaccurate outputs.

#### Pain Point 4: The Bilingual Bridge
Source laws are published exclusively in Nepali, while professional reporting and client communication are in English. Mental translation of precise legal terminology — under time pressure — is error-prone and cognitively exhausting.

#### Pain Point 5: The Finance Act Annual Churn
Every year on Jestha 15, the Finance Act amends multiple other Acts simultaneously. Manually tracking which provisions have been modified — and cascading those changes into advice — is a recurring annual crisis for every CA firm in Nepal.

### 2.2 Problem Validation

| Metric | Current State | Impact |
|---|---|---|
| Avg. research time per query | 3–4 hours (junior CA) | Billable hours lost |
| Regulatory update tracking | Manual / ad hoc | High miss rate |
| Table extraction accuracy (generic OCR) | 60–70% | Wrong tax calculations |
| Language barrier overhead | Constant | Fatigue + errors |
| Finance Act amendment tracking | Manual spreadsheet | Annual compliance crisis |

---

## 3. Goals & Success Metrics

### 3.1 Product Goals

- Reduce average compliance research time from hours to seconds via AI-powered retrieval.
- Eliminate regulatory hierarchy errors by always surfacing the most recent controlling authority.
- Achieve near-perfect table extraction accuracy on Nepali government PDFs using advanced IDP.
- Deliver bilingual (Nepali-source → English-output) reasoning without loss of legal precision.
- Provide verifiable, citation-backed answers — so professionals never have to "trust" the AI blindly.

### 3.2 Success Metrics (KPIs)

| KPI | Target (Month 6) | Measurement Method |
|---|---|---|
| Average query response time | < 5 seconds | Platform logs |
| Table extraction accuracy | > 98% | CA-verified test suite |
| Answer citation accuracy | > 95% | Spot audits by CA |
| Pilot user NPS score | > 50 | In-app survey |
| CA firm pilot conversions | 5 firms by Month 3 | CRM tracking |
| Regulatory database freshness | < 48 hrs from gazette | Automated monitoring |

---

## 4. Target Users & Market

### 4.1 User Segments

#### Tier 1 — CA & Audit Firms *(Primary Beachhead)*
~1,000+ registered CA firms in Nepal, each with 5–50 staff members conducting compliance research daily. Lowest-friction entry point: pain is acute, decision-making is centralized, ROI is immediately measurable in saved staff hours.

- **Primary user:** Junior CAs and audit associates performing document research
- **Economic buyer:** Managing Partner or firm principal
- **Estimated addressable firms:** 1,000+ (targeting top 200 in Phase 1)

#### Tier 2 — BFSI Sector *(High-Value Expansion)*
Nepal's banking, financial services, and insurance sector is legally mandated to monitor every NRB and Beema Samiti directive. Non-compliance carries heavy penalties. Highest willingness-to-pay and most structured compliance departments.

- 20+ commercial banks, 17 development banks, 50+ finance companies regulated by NRB
- 15+ insurance companies regulated by Beema Samiti
- **Primary user:** Chief Compliance Officers, compliance analysts, legal teams

#### Tier 3 — Large Corporates & MNCs
150+ hydropower companies, major manufacturing houses (Golchha Group, Chaudhary Group), and MNCs (Ncell, Unilever Nepal) operating under complex, multi-layered tax law.

### 4.2 User Personas

| Attribute | Persona A: Rohan (Junior CA) | Persona B: Sita (Head of Compliance, Bank) |
|---|---|---|
| **Role** | Audit Associate, CA firm | Chief Compliance Officer, Commercial Bank |
| **Daily task** | Research tax provisions, draft memos | Monitor NRB circulars, train staff |
| **Core frustration** | Spends 3–4 hrs/day on Gazette PDFs | Fears missing a new directive |
| **Success looks like** | Accurate answers in seconds with source | Automated alert when new circular drops |
| **Willingness to pay** | Firm-level subscription (medium) | Enterprise contract (high) |

---

## 5. Solution Overview & Feature Requirements

### 5.1 Product Architecture

Smart Finance Compliance is built on four architectural pillars, each directly addressing a specific pain point:

| Pillar | Technology | Pain Point Addressed |
|---|---|---|
| **Docsumo-Grade IDP** | Advanced Intelligent Document Processing pipeline with table-aware layout analysis | The "Ctrl+F" Void + Table Nightmare |
| **Hierarchy-Aware RAG** | Vector database with regulatory priority weighting; circulars override acts at retrieval time | Hierarchy Trap + Finance Act Churn |
| **Bilingual Reasoning Engine** | Cross-lingual embeddings (NepBERTa + multilingual models) | Bilingual Bridge |
| **Rajpatra Scraper & Indexer** | Automated crawler with < 48-hour freshness guarantee | Annual Finance Act Churn + real-time circulars |

### 5.2 Feature Requirements

#### F1 — Intelligent Query Interface `Must Have`
- Natural language query input in English or Nepali
- Instant response with synthesized answer, regulatory hierarchy path, and exact source citation
- Source citation links to the specific page of the government PDF
- Confidence score displayed for each answer
- Follow-up question chaining within a conversation thread

#### F2 — Document Intelligence Engine `Must Have`
- Ingest scanned Nepali PDF documents (Acts, Rules, Gazette notifications, circulars)
- Table extraction with ≥ 98% accuracy using layout-aware OCR models
- Named entity recognition tuned to Nepali legal terminology (section numbers, Bikram Sambat fiscal years)
- Automatic deduplication and versioning — newer documents override older ones of the same type

#### F3 — Regulatory Hierarchy Engine `Must Have`
- Hardcoded priority ladder: **Circular/Directive > Finance Act amendment > Principal Act > Rules**
- When a query retrieves conflicting provisions, surface the most recently controlling authority first
- Display the full hierarchy chain for each answer (e.g., *"Overridden by NRB Circular 2081/03"*)

#### F4 — Bilingual Output Engine `Must Have`
- User query in English triggers retrieval from Nepali source documents
- Synthesized answer returned in English with key Nepali legal terms preserved in parentheses
- Option to toggle to the raw Nepali source text for verification

#### F5 — Regulatory Alert & Monitoring Feed `Should Have`
- Automated daily scraping of Nepal Gazette, NRB website, and IRD portal
- Push notification (email + in-app) when a new directive is published in a subscribed domain
- AI-generated summary of the new document's key changes
- Configurable alert preferences per user role

#### F6 — Team Workspace & Audit Trail `Should Have`
- Multi-user workspace for firm-level subscriptions
- Saved queries and answers with researcher attribution
- Full audit log of who queried what and when (demonstrates due diligence)
- Shareable annotated answer reports exportable to PDF

#### F7 — Finance Act Amendment Tracker `Nice to Have — v1.1`
- Annual Jestha 15 Finance Act impact dashboard
- Side-by-side comparison of amended vs. original provision
- AI-generated impact summary: *"This amendment increases the TDS rate on rent from 10% to 15%."*

---

## 6. Non-Functional Requirements

| Category | Requirement | Specification |
|---|---|---|
| Performance | Query latency | P95 < 5 seconds end-to-end |
| Performance | Document ingestion | < 30 minutes per new Gazette issue |
| Accuracy | Table extraction | ≥ 98% field-level accuracy on test set |
| Accuracy | Citation correctness | ≥ 95% correct source page referenced |
| Availability | Platform uptime | ≥ 99.5% monthly SLA |
| Security | Data storage | Nepal-compliant cloud; no user query data shared with third parties |
| Scalability | Concurrent users | 500 at MVP; 5,000 by Year 1 |
| Compliance | Data handling | GDPR-aligned; no PII stored in vector database |

---

## 7. Competitive Analysis

### 7.1 Landscape Overview

The competitive landscape in Nepal is fragmented and underdeveloped. The primary competitor is not a software product — it is the entrenched manual research workflow itself.

| Competitor | Table OCR | NRB Circulars | Bilingual | Key Weakness |
|---|---|---|---|---|
| Manual research (status quo) | N/A | Manual | Manual | Slow, error-prone, not scalable |
| Eksana (legal) | Weak | No | No | Court cases only; no accounting/tax focus |
| ChatGPT / Claude (general) | Moderate | No | Partial | No 2081 updates; hallucinations on tax rates |
| Nepal Law Commission site | None | No | No | Basic keyword search; no AI reasoning |
| **Smart Finance Compliance** | **Industry-best** | **Yes (automated)** | **Yes (native)** | — |

### 7.2 Competitive Moat

- **The Table Extraction Moat:** In financial compliance, if the tool cannot correctly read a rate table, it is worse than useless. Our IDP pipeline is purpose-built for Nepali government table layouts — a capability no competitor has matched.

- **The CA-Verified Logic Moat:** Our CA domain expert ensures the hierarchy logic (which law overrides which) is correctly programmed. This is a domain knowledge problem our team uniquely solves.

- **The Speed-to-Law Moat:** Our Rajpatra scraper indexes new gazette publications within 48 hours. Competitors and general LLMs are months or years behind.

- **The Bilingual Embedding Moat:** Cross-lingual models trained on Nepali legal text are expensive to build. Once built, they create a durable data and model advantage.

---

## 8. Go-to-Market Strategy

### 8.1 Pricing Model

| Tier | Target Segment | Price (Annual) | Key Inclusions |
|---|---|---|---|
| **Starter** | CA firms < 10 staff | रू 36,000/yr | 3 seats, 500 queries/mo, core features |
| **Professional** | CA firms 10–50 staff | रू 1,20,000/yr | 15 seats, unlimited queries, alert feed, audit trail |
| **Enterprise** | Banks, Insurance, Corporates | Custom (रू 5L+) | Unlimited seats, custom domains, dedicated support, API access |

### 8.2 Launch Phases

**Phase 0 — Validation (Weeks 1–4)**
Recruit 2 CA collaborators. Build a functional demo covering the Income Tax Act and 3 recent IRD circulars. Present to 10 target CA firms and collect qualitative feedback.

**Phase 1 — Closed Beta (Months 2–3)**
Onboard 5 pilot CA firms at zero cost. Build feedback loop with domain experts. Prioritize fixing table extraction and hierarchy logic errors surfaced by real users.

**Phase 2 — Paid Launch (Months 4–6)**
Convert pilot firms to Starter/Professional plans. Expand document database to include NRB directives and Beema Samiti circulars for BFSI outreach.

**Phase 3 — Enterprise Expansion (Month 7+)**
Approach top 5 commercial banks. Launch Finance Act Amendment Tracker (F7). Introduce API tier for large corporate integrations.

---

## 9. Development Milestones & Roadmap

| Phase | Timeline | Key Deliverables | Status |
|---|---|---|---|
| Phase 0 | Weeks 1–4 | CA collaborator secured; IDP pipeline prototype; demo on 3 Acts | 🟡 Planning |
| Phase 1 | Month 2–3 | Full RAG pipeline; hierarchy engine; bilingual output; 5 pilot firms | ⬜ Not started |
| Phase 2 | Month 4–6 | Alert monitoring feed; team workspace; paid billing integration | ⬜ Not started |
| Phase 3 | Month 7–12 | Finance Act tracker; NRB circular domain; enterprise API; 20 paying firms | ⬜ Not started |

---

## 10. Risks & Mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | AI hallucination on tax rates | 🔴 Critical | Every numerical answer requires a verbatim source quote from the original PDF displayed alongside the synthesized response. No number shown without its citation. |
| 2 | Table extraction failure on complex layouts | 🔴 High | Maintain a human-verified test suite of 50 known tables. Run regression before each deployment. Flag low-confidence extractions for manual review. |
| 3 | Nepal Gazette changes publication format | 🟡 Medium | Build format-agnostic parsing with version detection. Monitor scraper failures weekly. |
| 4 | Low adoption due to trust barrier | 🔴 High | Lead with source citation UX. Let CAs verify every answer instantly. Run "spot-the-difference" demos against manual research. |
| 5 | CA regulatory liability concerns | 🟡 Medium | Clear disclaimer: platform is a research aid, not legal advice. Position as an accelerator for professional judgment, not a replacement. |
| 6 | Competitor enters with more capital | 🟢 Low (near-term) | Speed moat: go-to-market before any well-funded competitor builds Nepal-specific training data. Data network effect grows with each user query. |

---

## 11. Cost & Resource Estimates

### 11.1 Prototype Budget (Month 1)

| Item | Estimated Cost | Notes |
|---|---|---|
| LLM API costs (OpenAI / Anthropic) | रू 5,000 | For prototype query volume |
| Vector database (Pinecone / Weaviate) | रू 2,000 | Free tier sufficient initially |
| Cloud hosting (AWS / GCP) | रू 4,000 | t3.medium equivalent |
| IDP / OCR tooling | रू 3,000 | Docsumo trial or open-source |
| Miscellaneous (domains, tools) | रू 1,000 | — |
| **TOTAL** | **रू 15,000** | **~USD $115** |

> *Developer cost: Primary asset is the ML Engineer founder's time (~4 weeks of focused effort). No external engineering hires required at prototype stage.*

### 11.2 Year 1 Unit Economics (Projection)

| Metric | Conservative | Target |
|---|---|---|
| Paying firms (end of Year 1) | 15 firms | 40 firms |
| Average Annual Revenue per Firm | रू 60,000 | रू 90,000 |
| Gross Annual Revenue | रू 9,00,000 | रू 36,00,000 |
| Infrastructure & API Cost | रू 1,50,000 | रू 4,00,000 |
| **Gross Margin** | **~83%** | **~89%** |

---

## 12. Open Questions & Dependencies

### 12.1 Open Questions

- What is the most effective Nepal-specific cross-lingual embedding model — NepBERTa fine-tuned vs. multilingual-E5-large?
- Will CA firms require on-premise deployment for data confidentiality, or is cloud acceptable for financial documents?
- What is the minimum viable document corpus for the pilot to be credible? *(Hypothesis: Income Tax Act 2058 + 5 recent IRD circulars + NRB BFIS Directives.)*
- Is there an existing Nepal Law Commission or government API for gazette data, or must all ingestion be scraper-based?
- What liability disclaimers are legally required in Nepal for an AI-assisted legal/tax research tool?

### 12.2 Key Dependencies

- **CA Domain Expert:** At least one practicing CA must be engaged before Phase 1 to validate the hierarchy logic engine.
- **Nepal Gazette Access:** Reliable and consistent access to Rajpatra.gov.np for automated scraping.
- **IDP Model Quality:** The accuracy of the table extraction pipeline is the single largest technical risk and must be validated on real Nepali government PDFs in the first two weeks.
- **Pilot Firm Commitment:** 5 CA firms must agree to pilot the product during Phase 1. Outreach should begin during Phase 0.

---

## 13. Appendix: Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — an AI architecture that grounds LLM responses in a curated document database rather than model weights alone. |
| **IDP** | Intelligent Document Processing — advanced pipeline combining OCR, layout analysis, and ML to extract structured data from unstructured documents. |
| **NRB** | Nepal Rastra Bank — Nepal's central bank. Issues binding directives on all commercial and development banks. |
| **IRD** | Inland Revenue Department — Nepal's primary tax administration body. Issues circulars interpreting tax law. |
| **Nepal Gazette (Rajpatra)** | The official government publication in which all Acts, Rules, notifications, and amendments are legally promulgated. |
| **NFRS** | Nepal Financial Reporting Standards — Nepal's adaptation of IFRS for financial reporting. |
| **Beema Samiti** | Insurance Board of Nepal — regulatory authority for the insurance sector. |
| **BFSI** | Banking, Financial Services, and Insurance — collective term for Nepal's regulated financial sector. |
| **Finance Act (Artha Ain)** | Annual Act passed by Parliament every Jestha 15 that amends tax rates, exemptions, and provisions across multiple principal Acts. |
| **Bikram Sambat (BS)** | Nepal's official calendar. Fiscal Year 2081 BS corresponds to 2024/25 AD. |

---

*End of Document — Smart Finance Compliance PRD v1.0 | CONFIDENTIAL*
