package main

import "github.com/bornholm/indecis"

// classic est la liste de catégories de la preuve de concept : un tri
// classique de boîte de réception d'organisation. Elle ne sert jamais à
// l'entraînement, pour mesurer ce que vaut une liste inconnue du modèle.
var classic = []indecis.Candidate{
	{Name: "Support technique", Description: "Panne, bug, problème d'accès ou demande d'aide sur un outil, un logiciel ou un service informatique."},
	{Name: "Facturation et paiement", Description: "Factures, paiements, relances, remboursements, questions de tarif sur une prestation déjà vendue."},
	{Name: "Commande et livraison", Description: "Suivi de commande, expédition, retard de livraison, retour ou échange de produit."},
	{Name: "Réclamation", Description: "Mécontentement exprimé ou plainte formelle d'un client ou d'un partenaire."},
	{Name: "Demande commerciale", Description: "Demande de devis, de tarifs ou d'informations avant achat, proposition commerciale, prospection."},
	{Name: "Réunion et agenda", Description: "Invitation, planification, report ou annulation de réunion, compte rendu de réunion."},
	{Name: "Ressources humaines", Description: "Recrutement, candidature, congés, paie, formation, contrat de travail, vie des salariés."},
	{Name: "Juridique et contrats", Description: "Contrats, accords, clauses, conformité réglementaire, litiges, avis juridiques."},
	{Name: "Suivi de projet", Description: "Échanges de travail sur un projet ou une activité : avancement, rapports, chiffres, documents à relire."},
	{Name: "Newsletter et marketing", Description: "Lettre d'information, promotion, annonce ou communication envoyée à une liste de diffusion."},
	{Name: "Notification automatique", Description: "Message généré par un système : alerte, confirmation, accusé de réception, rappel automatique."},
	{Name: "Personnel", Description: "Échange privé, non professionnel : famille, amis, loisirs."},
	{Name: "Spam et hameçonnage", Description: "Message non sollicité, publicité abusive ou tentative de fraude."},
}

// taxonomy est une liste de catégories d'entraînement. Les listes varient
// par le point de vue (action attendue, domaine, relation, genre, ton), la
// langue et le niveau de détail : le modèle doit apprendre à lire une liste,
// pas à en retenir une. Aucune ne reprend la liste classique, réservée au
// test.
type taxonomy struct {
	Name     string // nom de question : [a-z][a-z0-9_]*
	Question string
	Options  []indecis.Candidate
}

var training = []taxonomy{
	{"action", "What does the recipient need to do with this email?", []indecis.Candidate{
		{Name: "Reply needed", Description: "The sender expects an answer from the recipient."},
		{Name: "Approval requested", Description: "The recipient must approve, sign off or authorize something."},
		{Name: "Task assigned", Description: "The recipient is asked to do a piece of work."},
		{Name: "Scheduling", Description: "Find a time, accept or move a meeting or a call."},
		{Name: "FYI only", Description: "Information shared, no action expected."},
	}},
	{"domaine", "À quel domaine d'activité le courriel se rattache-t-il ?", []indecis.Candidate{
		{Name: "Marchés et négoce", Description: "Achat et vente d'énergie, prix, positions, contrats de marché."},
		{Name: "Régulation", Description: "Autorités de régulation, lois, audiences, conformité."},
		{Name: "Finance et comptabilité", Description: "Budgets, comptes, trésorerie, résultats financiers."},
		{Name: "Informatique", Description: "Systèmes, logiciels, accès, matériel."},
		{Name: "Ressources humaines", Description: "Personnel, recrutement, évaluations, avantages."},
		{Name: "Communication interne", Description: "Annonces de l'entreprise, organisation, événements internes."},
		{Name: "Relations clients", Description: "Échanges avec des clients ou des prospects."},
		{Name: "Vie privée", Description: "Sujets personnels sans lien avec le travail."},
	}},
	{"relation", "Who is the sender relative to the recipient?", []indecis.Candidate{
		{Name: "Colleague"}, {Name: "Manager or executive"}, {Name: "External partner"},
		{Name: "Customer"}, {Name: "Vendor or supplier"}, {Name: "Automated system"}, {Name: "Friend or family"},
	}},
	{"genre", "What kind of message is it?", []indecis.Candidate{
		{Name: "Question"}, {Name: "Announcement"}, {Name: "Report or data"}, {Name: "Request for help"},
		{Name: "Social invitation"}, {Name: "Thanks or congratulations"}, {Name: "Complaint"}, {Name: "Newsletter or digest"},
	}},
	{"sujet", "Quel est le sujet principal ?", []indecis.Candidate{
		{Name: "Contrats"}, {Name: "Réunions"}, {Name: "Déplacements"}, {Name: "Budget"},
		{Name: "Recrutement"}, {Name: "Sport et loisirs"}, {Name: "Politique et actualité"}, {Name: "Technologie"},
		{Name: "Gaz et électricité"},
	}},
	{"urgence", "Quelle urgence pour le destinataire ?", []indecis.Candidate{
		{Name: "Urgent", Description: "À traiter aujourd'hui."},
		{Name: "Cette semaine", Description: "À traiter dans les prochains jours."},
		{Name: "Sans échéance", Description: "Aucun délai particulier."},
	}},
	{"tone", "What is the tone of the email?", []indecis.Candidate{
		{Name: "Friendly"}, {Name: "Neutral and factual"}, {Name: "Formal"}, {Name: "Frustrated or negative"}, {Name: "Humorous"},
	}},
	{"archive", "Dans quel dossier ranger ce courriel ?", []indecis.Candidate{
		{Name: "Clients"}, {Name: "Fournisseurs"}, {Name: "Interne"}, {Name: "Administratif"},
		{Name: "Personnel"}, {Name: "Lettres d'information"}, {Name: "À supprimer", Description: "Sans intérêt, publicité, message automatique périmé."},
	}},
}

// teacherSchema écrit le schéma et les consignes des teachers pour les
// listes d'entraînement.
func teacherSchema() (indecis.Schema, string) {
	var s indecis.Schema
	g := "# Classement de courriels selon plusieurs listes\n\nPour chaque liste, choisis UNE catégorie : celle qui décrit le mieux le courriel, du point de vue de la personne qui le reçoit. Les courriels viennent d'Enron, une entreprise d'énergie (2000-2002).\n"
	for _, t := range training {
		var names []string
		g += "\n## " + t.Name + " — " + t.Question + "\n\n"
		for _, o := range t.Options {
			names = append(names, o.Name)
			if o.Description != "" {
				g += "- " + o.Name + " : " + o.Description + "\n"
			} else {
				g += "- " + o.Name + "\n"
			}
		}
		s = append(s, indecis.NewChoice(t.Name, t.Question, names...))
	}
	return s, g
}

// templated sont les listes des courriels synthétiques en français
// (synth/courriel) : leurs étiquettes viennent des gabarits. Aucun nom ne
// reprend la liste classique.
var templated = []taxonomy{
	{"service", "Quel service est concerné ?", names("Comptabilité", "Paie et personnel", "Informatique interne", "Achats",
		"Service juridique", "Logistique", "Relation client", "Direction", "Communication", "Vie d'équipe")},
	{"intention", "Quelle est l'intention de l'expéditeur ?", names("Demande d'information", "Relance", "Confirmation", "Plainte",
		"Invitation", "Transmission de document", "Demande d'intervention", "Demande d'accord", "Remerciement", "Annulation", "Alerte", "Information")},
	{"emetteur", "Qui écrit ?", names("Client", "Fournisseur", "Collègue", "Hiérarchie", "Candidat", "Administration", "Système automatique")},
	{"delai", "Dans quel délai répondre ?", names("Immédiat", "Sous quelques jours", "Aucun")},
	{"registre", "Quel registre de langue ?", names("Soutenu", "Courant", "Familier", "Impersonnel")},
}

func names(ns ...string) []indecis.Candidate {
	out := make([]indecis.Candidate, len(ns))
	for i, n := range ns {
		out[i] = indecis.Candidate{Name: n}
	}
	return out
}
